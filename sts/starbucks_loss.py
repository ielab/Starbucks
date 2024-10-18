import random
import warnings
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.models import Transformer
from matryoshkaloss_modified import MatryoshkaLoss


class StarbucksLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_layers: list[int],
        matryoshka_dims: list[int],
        matryoshka_weights: list[float | int] | None = None,
        n_selections_per_step: int = -1,
        last_layer_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ) -> None:
        """
        The StarbucksLoss can be seen as a loss *modifier* that allows you to use other loss functions at non-final
        layers of the Sentence Transformer model. This is useful for when you want to train a model where users have
        the option to train on a set of layer-dimensionality pairs. The StarbucksLoss allows you to train on these pairs

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            matryoshka_layers: The layers to use for the loss. The layers
                are 1-indexed, so the first layer is 1, the second layer
                is 2, etc. Example is `[2, 4, 6, 8, 10, 12]`.
            matryoshka_dims: The dimensions to use for the loss.
                Example is `[32, 64, 128, 256, 512, 768]`.
            matryoshka_weights: The weights to use for the loss of each layer-dimensionality pair.
            n_selections_per_step: The number of layers to use per step. If
                -1, then all layers are used. If > 0, then a random
                sample of `n_layers_per_step` layers are used per step,
                separate from the final layer, which is always used. The
                2DMSE paper uses `n_layers_per_step=1`. The default
                value is 1.
            last_layer_weight: The weight to use for the loss of the
                final layer. Increase this to focus more on the
                performance when using all layers. The default value is
                1.0.
            kl_div_weight: The weight to use for the KL-divergence loss
                that is used to make the prior layers match that of the
                last layer. Increase this to focus more on the
                performance when using fewer layers. The default value
                is 1.0.
            kl_temperature: The temperature to use for the KL-divergence
                loss. If 0, then the KL-divergence loss is not used. The
                default value is 1.0.

        References:
            - The concept was inspired by the Starbucks paper: https://arxiv.org/pdf/2410.13230v1

        Requirements:
            1. The base loss cannot be :class:`CachedMultipleNegativesRankingLoss` or :class:`CachedGISTEmbedLoss`.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+
        """
        super().__init__()
        self.model = model
        self.loss = MatryoshkaLoss(model, loss, matryoshka_dims,
            matryoshka_weights=matryoshka_weights,
            n_dims_per_step=n_selections_per_step,)

        self.n_selections_per_step = n_selections_per_step
        # the number of dim and layers should be the same
        assert len(matryoshka_layers) == len(matryoshka_dims)
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_layers = [layer-1 for layer in matryoshka_layers]
        self.last_layer_weight = last_layer_weight
        self.prior_layers_weight = 1
        self.kl_div_weight = kl_div_weight
        self.kl_temperature = kl_temperature
        assert isinstance(self.model[0], Transformer)
        if isinstance(loss, CachedMultipleNegativesRankingLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedMultipleNegativesRankingLoss.", stacklevel=2)
        if isinstance(loss, CachedGISTEmbedLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedGISTEmbedLoss.", stacklevel=2)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Decorate the forward function of the transformer to cache the embeddings of all layers
        original_transformer_forward = self.model[0].forward
        transformer_decorator = TransformerDecorator(self.model[0], original_transformer_forward)
        self.model[0].forward = transformer_decorator

        # Decorate the forward function of the model to get the embeddings after all modules (e.g. pooling)
        original_forward = self.model.forward
        forward_decorator = ForwardDecorator(original_forward)

        self.model.forward = forward_decorator

        # Run the loss normally: i.e. the final layer, but 1) use the transformers decorator to cache
        # the embeddings of all layers and 2) use the forward decorator to get the embeddings after all modules
        # for the KL-divergence loss
        loss = self.loss(sentence_features, labels, 768) * self.last_layer_weight
        if self.kl_temperature > 0:
            final_embeddings = forward_decorator.get_embeddings()

        num_layers = transformer_decorator.num_layers

        # remove the last layer, as we already computed the loss over it
        layer_indices = [i for i in self.matryoshka_layers if i < num_layers - 1]
        dim_indices = [self.matryoshka_dims[i] for i in range(len(layer_indices))]

        if self.n_selections_per_step > 0 and self.n_selections_per_step < num_layers - 1:
            layer_indices = random.sample(layer_indices, self.n_selections_per_step)
            #dim should be based on the index of the layer
            dim_indices = [self.matryoshka_dims[layer_indices.index(i)] for i in layer_indices]

        # This loop is over `num_layer - 1` layers because we already computed the loss over the final layer
        for layer_idx, dim_idx in zip(layer_indices, dim_indices):
            #
            # Add regular loss for each layer by using the cached embeddings of that layer
            transformer_decorator.set_layer_idx(layer_idx)
            layer_loss = self.loss(sentence_features, labels, dim_idx)
            loss = loss + layer_loss * self.prior_layers_weight

            # and KL-divergence loss between the current layer and the final layer
            # Note: we use "batchmean" reduction as that aligns with the mathematical definition
            if self.kl_temperature > 0:
                embeddings = forward_decorator.get_embeddings()
                # copy a final embeddings with the same shape as embeddings by cutting dimensions
                kl_div_loss = self.kl_divergence_loss(embeddings, final_embeddings.clone(), dim=dim_idx, reduction="batchmean")
                loss = loss + kl_div_loss * self.kl_temperature * self.kl_div_weight

        self.model[0].forward = original_transformer_forward
        self.model.forward = original_forward

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "n_selections_per_step": self.n_selections_per_step,
            "last_layer_weight": self.last_layer_weight,
            "prior_layers_weight": self.prior_layers_weight,
            "kl_div_weight": self.kl_div_weight,
            "kl_temperature": self.kl_temperature,
            "matryoshka_layers": self.matryoshka_layers,
            "matryoshka_dims": self.matryoshka_dims,
        }
    #
    def kl_divergence_loss(self, p, q, dim =-1, reduction="batchmean") -> Tensor:
        """
        Compute the KL-divergence between two distributions.

        Args:
            p (Tensor): The first distribution. The KL-divergence is computed as `KL(p || q)`.
            q (Tensor): The second distribution.
            reduction (str, optional): Specifies the reduction to apply to the output. Default: "mean".

        Returns:
            Tensor: The KL-divergence between `p` and `q`.
        """
        if dim > 0:
            p = p[:, :dim]
            q = q[:, :dim]
        return F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction=reduction)


class TransformerDecorator:
    """
    Decorator that caches the embeddings of all layers of the transformer.
    When `layer_idx` is set, it returns the cached embeddings of that layer instead.

    This is meant to override the forward function of the Transformer.
    """

    def __init__(self, transformer: Transformer, original_forward) -> None:
        self.transformer = transformer
        self.original_forward = original_forward
        self.embeddings: list[tuple[Tensor]] = []
        self.last_embeddings: list[Tensor] = []
        self.features: list[dict[str, Tensor]] = []
        self.layer_idx = None
        self.call_idx = 0

    def set_layer_idx(self, layer_idx) -> None:
        self.layer_idx = layer_idx
        self.call_idx = 0

    def get_layer_embeddings(self) -> Tensor:
        return torch.concat([embedding[self.layer_idx] for embedding in self.embeddings], dim=1)

    def __call__(self, features) -> dict[str, Tensor]:
        if self.layer_idx is None:
            output = self.call_grow_cache(features)
        else:
            output = self.call_use_cache(features)
            self.call_idx += 1
        return output

    def call_grow_cache(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Temporarily sets the output_hidden_states to True, runs the model, and then restores the original setting.
        Use the all_layer_embeddings to get the embeddings of all layers.
        """
        original_output_hidden_states = self.transformer.auto_model.config.output_hidden_states
        self.transformer.auto_model.config.output_hidden_states = True

        output = self.original_forward(features)
        # We ignore the first layer, as it is the input embeddings
        # and the last layer, as we already computed the loss over it
        self.num_layers = len(output["all_layer_embeddings"]) - 1
        self.embeddings.append(output["all_layer_embeddings"][1:-1])
        self.last_embeddings.append(output["token_embeddings"])
        self.features.append(
            {key: value for key, value in output.items() if key not in ["all_layer_embeddings", "token_embeddings"]}
        )

        # Restore original setting
        self.transformer.auto_model.config.output_hidden_states = original_output_hidden_states

        if original_output_hidden_states:
            del output["all_layer_embeddings"]

        return output

    def call_use_cache(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        return {**self.features[self.call_idx], "token_embeddings": self.embeddings[self.call_idx][self.layer_idx]}




class ForwardDecorator:
    """
    Decorator that caches the embeddings after all modules (e.g. pooling) of the model.
    Required to get the embeddings after all modules for the KL-divergence loss.

    This is meant to override the forward function of the SentenceTransformer.
    """

    def __init__(self, fn) -> None:
        self.fn = fn
        self.embeddings = []

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.fn(features)
        self.embeddings.append(output["sentence_embedding"])
        return output

    def get_embeddings(self) -> Tensor:
        embeddings = torch.concat(self.embeddings, dim=0)
        self.embeddings = []
        return embeddings