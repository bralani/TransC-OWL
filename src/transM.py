import logging
import math
import os
import time
from class_resolver import OneOrManyHintOrType, OneOrManyOptionalKwargs
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.training import SLCWATrainingLoop
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import kl_divergence
from pykeen.models import ERModel
from torch import nn
from pykeen.typing import HeadRepresentation, InductiveMode, RelationRepresentation, TailRepresentation, COLUMN_HEAD, COLUMN_TAIL, GaussianDistribution
from pykeen.triples import KGInfo
from pykeen.regularizers import LpRegularizer, NoRegularizer
from pykeen.nn.modules import FunctionalInteraction
from pykeen.nn import *
from pykeen.models import Model
from pykeen.losses import Loss, MarginRankingLoss
from torch.optim import SGD, AdamW, Adam
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.utils import check_shapes
from pykeen.training import TrainingCallback
from art.train import art_train
from typing import (
    Any,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast
)
logger = logging.getLogger(__name__)

subclassof_id = 0
typeof_id = 1
relations_ids = []
entity_ids = []
mapped_triples = []
entity_label_ids = []
classes_ids = []
classes_label_ids = []

# preleva il nome del dataset da terminale
dataset = input("Inserisci il nome del dataset: ")
# se non ha inserito niente, usa il dataset di default
if dataset == "":
    dataset = "DBpedia100K"
# preleva embedding_dim da terminale
embedding_dim = int(input("Inserisci la dimensione degli embedding: "))
# se non ha inserito niente, usa il valore di default   
if embedding_dim == "":
    embedding_dim = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

class BatchSaveModel(TrainingCallback):
    def post_epoch(self, epoch: int, epoch_loss) -> None:

        if is_grid_search == False:
            if epoch % 50 == 0 or epoch == 1:
                # Converti il tensore in un array NumPy
                entity_embedding = self.model.entity_representations[0](indices=None).clone().detach().cpu().numpy()
                relation_embedding = self.model.relation_representations[0](indices=None).clone().detach().cpu().numpy()

                class_embedding_mu = self.model.class_representations[0](indices=None).clone().detach().cpu().numpy()
                class_embedding_cov = self.model.class_representations[1](indices=None).clone().detach().cpu().numpy()

                # Salva gli embedding in un file di testo
                np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/entity2vec_'+str(epoch)+'.txt', entity_embedding)
                np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/relation2vec_'+str(epoch)+'.txt', relation_embedding)
                np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/classmu2vec_'+str(epoch)+'.txt', class_embedding_mu)
                np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/classcov2vec_'+str(epoch)+'.txt', class_embedding_cov)
        elif epoch == 20:
            # Converti il tensore in un array NumPy
            entity_embedding = self.model.entity_representations[0](indices=None).clone().detach().cpu().numpy()
            relation_embedding = self.model.relation_representations[0](indices=None).clone().detach().cpu().numpy()

            class_embedding_mu = self.model.class_representations[0](indices=None).clone().detach().cpu().numpy()
            class_embedding_cov = self.model.class_representations[1](indices=None).clone().detach().cpu().numpy()

            # crea la cartella se non esiste
            if not os.path.exists('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof)):
                os.makedirs('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof))

            # Salva gli embedding in un file di testo
            np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof) + '/entity2vec.txt', entity_embedding)
            np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof) + '/relation2vec.txt', relation_embedding)
            np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof) + '/classmu2vec.txt', class_embedding_mu)
            np.savetxt('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/'+str(self.model.margin)+"/"+str(self.model.margin_typeof)+"/"+str(self.model.margin_subclassof) + '/classcov2vec.txt', class_embedding_cov)

    

def _prepare_representation_module_list(
    max_id: int,
    shapes: Sequence[str],
    label: str,
    representations: OneOrManyHintOrType[Representation] = None,
    representations_kwargs: OneOrManyOptionalKwargs = None,
    skip_checks: bool = False,
) -> Sequence[Representation]:
    rs = representation_resolver.make_many(representations, kwargs=representations_kwargs, max_id=max_id)

    # check max-id
    for r in rs:
        if r.max_id < max_id:
            raise ValueError(
                f"{r} only provides {r.max_id} {label} representations, but should provide {max_id}.",
            )
        elif r.max_id > max_id:
            logger.warning(
                f"{r} provides {r.max_id} {label} representations, although only {max_id} are needed."
                f"While this is not necessarily wrong, it can indicate an error where the number of {label} "
                f"representations was chosen wrong.",
            )

    rs = cast(Sequence[Representation], nn.ModuleList(rs))
    if skip_checks:
        return rs

    # check shapes
    if len(rs) != len(shapes):
        raise ValueError(
            f"Interaction function requires {len(shapes)} {label} representations, but {len(rs)} were given."
        )
    check_shapes(
        *zip(
            (r.shape for r in rs),
            shapes,
        ),
        raise_on_errors=True,
    )
    return rs

def mahalanobis_typeof(x_list, mean_list, covariance_list):
    prob = []
    
    for x, mean, cov in zip(x_list, mean_list, covariance_list):
        centered_point = x - mean
        inv_covariance = torch.inverse(torch.diag(cov))
        distance = torch.sqrt(torch.sum(centered_point**2 * inv_covariance))

        prob.append(distance)
    
    return torch.stack(prob)

def prob_typeof(x_list, mean_list, covariance_list):
    '''
    prob = []

    for x, mean, cov in zip(x_list, mean_list, covariance_list):
        #prob.append(torch.distributions.MultivariateNormal(mean, torch.diag(cov)).log_prob(x))
        diff = x - mean
        inv_cov = 1 / (torch.sqrt(cov) + 0.001)
        exponential_term = -0.5 * torch.einsum("i,i,i->", diff, inv_cov, diff)
        log_prob = -0.5 * torch.sum(torch.log(cov)) + exponential_term
        prob.append(log_prob)

    '''
    diff = x_list - mean_list
    inv_cov = 1 / (torch.sqrt(covariance_list) + 0.001)
    exponential_term = -0.5 * torch.einsum("ni,ni,ni->n", diff, inv_cov, diff)
    log_prob = -0.5 * torch.sum(torch.log(covariance_list), dim=1) + exponential_term

    '''


    # controlla se ci sono nan o inf
    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
        # prelva gli indici dei nan e inf
        nan_index = torch.where(torch.isnan(log_prob))
        inf_index = torch.where(torch.isinf(log_prob))

        inv
        print("nan or inf")
    diff = x_list - mean_list
    inv_covariance_matrix = 1.0 / covariance_list
    mahalanobis = torch.sqrt(torch.sum(torch.square(diff) * inv_covariance_matrix))
    scale_factor = torch.sqrt(torch.sum(covariance_list, dim=1))  # Costante di scala (deviazione standard)
    normalized_mahalanobis = mahalanobis / scale_factor

    # Imposta un fattore moltiplicativo per definire il raggio dell'ellissoide
    multiplier = 10  # Esempio: 2 corrisponde a un intervallo di confidenza del 95%

    # Calcola il raggio dell'ellissoide come il prodotto delle varianze e del fattore moltiplicativo
    ellipsoid_radius = torch.sqrt(covariance_list) * multiplier
    ellipsoid_radius = torch.mean(ellipsoid_radius, dim=1)

    scores = ellipsoid_radius - normalized_mahalanobis
    
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        # prelva gli indici dei nan e inf
        nan_index = torch.where(torch.isnan(scores))
        inf_index = torch.where(torch.isinf(scores))

        print("nan or inf")
    '''
    return log_prob

def typeof_score(
    h: torch.FloatTensor,
    t_mean: torch.FloatTensor,
    t_var: torch.FloatTensor,
) -> torch.FloatTensor:
    return prob_typeof(h, t_mean, t_var)

class TransKLInteractionTypeOf(
    FunctionalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    
    _tail_indices = (0,1)
    func = typeof_score
    
    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h,
        r,
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:
        t_mean, t_var = t
        return dict(
            h=h,
            t_mean=t_mean,
            t_var=t_var,
        )

def inclusion_subclassof(mu_h, cov_h, mu_t, cov_t):

    n = mu_h.shape[-1]

    # Calcola la differenza tra il punto x e il centro mu
    distance = torch.norm(mu_h - mu_t)

    # Calcola le dimensioni relative degli ellissoidi
    log_det_h = torch.sum(torch.log(cov_h), dim=1)
    volumes_h = 0.5 * n * torch.log(2 * torch.tensor(np.pi)) + 0.5 * log_det_h

    log_det_t = torch.sum(torch.log(cov_t), dim=1)
    volumes_t = 0.5 * n * torch.log(2 * torch.tensor(np.pi)) + 0.5 * log_det_t

    size_ratio = volumes_h - volumes_t

    # Calcola la metrica di contenimento
    metrics = distance + size_ratio

    return metrics


def subclass_score(
    h_mean: torch.FloatTensor,
    h_var: torch.FloatTensor,
    t_mean: torch.FloatTensor,
    t_var: torch.FloatTensor,
) -> torch.FloatTensor:
    score = -inclusion_subclassof(h_mean, h_var, t_mean, t_var)
    return score

class TransKLInteractionSubClassOf(
    FunctionalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):

    _head_indices = (0,1)
    _head_indices = (0,1)
    func = subclass_score

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: Tuple[torch.FloatTensor, torch.FloatTensor],
        r,
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:
        h_mean, h_var = h
        t_mean, t_var = t
        return dict(
            h_mean=h_mean,
            h_var=h_var,
            t_mean=t_mean,
            t_var=t_var,
        )

def random_replacement_(batch: torch.LongTensor, index: int, selection: slice, size: int, tipo) -> None:

        if(tipo == "entity"):
            max_index = len(entity_ids)
        else:
            max_index = len(classes_ids) - 1

        replacement = torch.randint(
            high=max_index - 1,
            size=(size,),
            device=batch.device,
        )
        replacement += (replacement >= batch[selection, index]).long()

        if(tipo == "class"):
            #sostituisci ogni occorrenza di replacement
            replacement.apply_(lambda x: entity_label_ids[classes_label_ids[x]])

        batch[selection, index] = replacement

class TransKLNegativeSampler(BernoulliNegativeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102

        # seleziona dal batch le triple che hanno come relazione subClassOf
        subClassOf = positive_batch[positive_batch[:, 1] == subclassof_id]
        if(len(subClassOf) > 0):
            negative_batch_sub = self.corrupt_batch_custom(subClassOf, "class", "class")
            batch_neg = negative_batch_sub

        # seleziona dal batch le triple che hanno come relazione typeOf
        typeOf = positive_batch[positive_batch[:, 1] == typeof_id]
        if(len(typeOf) > 0):
            negative_batch_type = self.corrupt_batch_custom(typeOf, "entity", "class")
            if(len(subClassOf) > 0):
                batch_neg = torch.cat([batch_neg, negative_batch_type])
            else:
                batch_neg = negative_batch_type


        # seleziona dal batch le triple che non sono né subClassOf né typeOf
        other = positive_batch[(positive_batch[:, 1] != subclassof_id) & (positive_batch[:, 1] != typeof_id)]
        if(len(other) > 0):
            negative_batch_other = super().corrupt_batch(other)
            if(len(subClassOf) > 0 or len(typeOf) > 0):
                batch_neg = torch.cat([batch_neg, negative_batch_other])
            else:
                batch_neg = negative_batch_other

        return batch_neg
    

    # docstr-coverage: inherited
    def corrupt_batch_custom(self, positive_batch: torch.LongTensor, head_type, tail_type) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # Decide whether to corrupt head or tail
        head_corruption_probability = self.corrupt_head_probability[positive_batch[..., 1]].unsqueeze(dim=-1)
        head_mask = torch.rand(
            *batch_shape, self.num_negs_per_pos, device=positive_batch.device
        ) < head_corruption_probability.to(device=positive_batch.device)

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
        # flatten mask
        head_mask = head_mask.view(-1)

        for index, mask in (
            (COLUMN_HEAD, head_mask),
            # Tails are corrupted if heads are not corrupted
            (COLUMN_TAIL, ~head_mask),
        ):
            random_replacement_(
                batch=negative_batch,
                index=index,
                tipo=head_type if index == COLUMN_HEAD else tail_type,
                selection=mask,
                size=mask.sum()
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)


class TransKL(ERModel):
    
    #: The class representations
    class_representations: Sequence[Representation]

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        embedding_dim: int = 100,
        scoring_fct_norm: int = 1,
        margin=1, margin_typeof=0.001, margin_subclassof=0.001,is_grid_search = False,
        regularizer,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations=Embedding,
            entity_representations_kwargs=dict(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
            ),
            relation_representations=Embedding,
            relation_representations_kwargs=dict(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
            ),
            **kwargs,
        )

        self.margin = margin
        self.margin_typeof = margin_typeof
        self.margin_subclassof = margin_subclassof
        self.is_grid_search = is_grid_search

        self.interaction_typeof = interaction_resolver.make(TransKLInteractionTypeOf)
        self.interaction_subclassOf = interaction_resolver.make(TransKLInteractionSubClassOf)
        self.triples_factory = triples_factory
        classe = 0
        self.dict_classes = {}
        for triple in triples_factory.triples:
            if triple[1] == 'typeof':
                if triple[2] not in self.dict_classes:
                    self.dict_classes[triple[2]] = classe
                    classe += 1
            elif triple[1] == 'subclassof':
                if triple[0] not in self.dict_classes:
                    self.dict_classes[triple[0]] = classe
                    classe += 1
                if triple[2] not in self.dict_classes:
                    self.dict_classes[triple[2]] = classe
                    classe += 1

        num_classes = len(self.dict_classes)

        global relations_ids
        relations_ids = self.triples_factory.relation_id_to_label
        global entity_ids
        entity_ids = self.triples_factory.entity_id_to_label
        global entity_label_ids
        entity_label_ids = self.triples_factory.entity_to_id
        global classes_ids
        classes_ids = self.dict_classes
        global classes_label_ids
        classes_label_ids = {v: k for k, v in self.dict_classes.items()}
        global mapped_triples
        mapped_triples = self.triples_factory.mapped_triples

        global subclassof_id
        subclassof_id = self.triples_factory.relation_to_id['subclassof']
        global typeof_id
        typeof_id = self.triples_factory.relation_to_id['typeof']

        self.class_representations = _prepare_representation_module_list(
            shapes=(num_classes, embedding_dim),
            max_id=num_classes,
            representations=Embedding,
            representations_kwargs=[
                # mean
                dict(
                    shape=embedding_dim,
                    initializer="xavier_normal_norm",
                ),
                # diagonal covariance
                dict(
                    shape= embedding_dim,
                    # Ensure positive definite covariances matrices and appropriate size by clamping
                    constrainer="clamp",
                    constrainer_kwargs=dict(min=0.001),
                    initializer="xavier_normal_norm",
                ),
            ],
            skip_checks=True,
            label="classes"
        )

        self.save_entity()
        self.save_relation()
        self.save_classes()

    def save_entity(self):

        file_out = open('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/entity_map.txt', 'w')

        # cicla sulla mappa delle entità
        for key, value in entity_label_ids.items():
            file_out.write(str(key) + ' ' + str(value) + '\n')

    def save_relation(self):
            
        file_out = open('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/relation_map.txt', 'w')

        # cicla sulla mappa delle relazioni
        for key, value in relations_ids.items():
            file_out.write(str(key) + ' ' + str(value) + '\n')

    def save_classes(self):

        file_out = open('D:/TransC-OWL/data/'+str(dataset)+'/Output/transprob/class_map.txt', 'w')

        # cicla sulla mappa delle classi
        for key, value in self.dict_classes.items():
            file_out.write(str(key) + ' ' + str(value) + '\n')

    def _get_representations_typeof(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails."""

        entity_to_id = self.triples_factory.entity_id_to_label
        tail_labels = [entity_to_id[int(chiave)] for chiave in t]
        t = torch.tensor([self.dict_classes[chiave] for chiave in tail_labels])

        head_representations = self._get_entity_representations_from_inductive_mode(mode=mode)
        tail_representations = self.class_representations
        head_representations = [head_representations[i] for i in self.interaction.head_indices()]
        tail_representations = [tail_representations[i] for i in self.interaction_typeof.tail_indices()]
        hr, rr, tr = [
            [representation(indices=indices) for representation in representations]
            for indices, representations in (
                (h, head_representations),
                (r, self.relation_representations),
                (t, tail_representations),
            )
        ]

        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )
    
    def _get_representations_subclassof(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails."""

        entity_to_id = self.triples_factory.entity_id_to_label
        head_labels = [entity_to_id[int(chiave)] for chiave in h]
        h = torch.tensor([self.dict_classes[chiave] for chiave in head_labels])
        tail_labels = [entity_to_id[int(chiave)] for chiave in t]
        t = torch.tensor([self.dict_classes[chiave] for chiave in tail_labels])

        head_representations = tail_representations = self.class_representations
        head_representations = tail_representations = [head_representations[i] for i in self.interaction_subclassOf.head_indices()]
        hr, rr, tr = [
            [representation(indices=indices) for representation in representations]
            for indices, representations in (
                (h, head_representations),
                (r, self.relation_representations),
                (t, tail_representations),
            )
        ]

        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )
    
    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None, type = "rel") -> torch.FloatTensor:

        if(type == "rel"):
            h,r,t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)

            score_rel = self.interaction.score_hrt(h=h, r=r, t=t)
            if score_rel.dim() == 1:
                score_rel = score_rel.unsqueeze(0)
            return score_rel
        
        if(type == "subclass"):
            h,r,t = self._get_representations_subclassof(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)

            score_sub = self.interaction_subclassOf.score_hrt(h=h, r=r, t=t)
            if score_sub.dim() == 1:
                score_sub = score_sub.unsqueeze(0)
            return score_sub
        
        if(type == "typeof"):
            h,r,t = self._get_representations_typeof(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)

            score_type = self.interaction_typeof.score_hrt(h=h, r=r, t=t)
            if score_type.dim() == 1:
                score_type = score_type.unsqueeze(0)

            return score_type


class TransKLLoss(MarginRankingLoss):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        positive_scores_typeof: torch.FloatTensor,
        negative_scores_typeof: torch.FloatTensor,
        positive_scores_subclassof: torch.FloatTensor,
        negative_scores_subclassof: torch.FloatTensor,
        margin, margin_typeof, margin_subclassof,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:
        
        self.margin = 1
        if (len(positive_scores)) > 0:
            loss_rel = super().process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=batch_filter,
                num_entities=num_entities,
            ) * margin
        else:
            loss_rel = torch.tensor(0.0)

        if (len(positive_scores_subclassof)) > 0:
            loss_subclassof = super().process_slcwa_scores(
                positive_scores=positive_scores_subclassof,
                negative_scores=negative_scores_subclassof,
                label_smoothing=label_smoothing,
                batch_filter=batch_filter,
                num_entities=num_entities,
            )
            loss_subclassof = loss_subclassof * margin_subclassof
        else:
            loss_subclassof = torch.tensor(0.0)

        if (len(positive_scores_typeof)) > 0:
            loss_typeof = super().process_slcwa_scores(
                positive_scores=positive_scores_typeof,
                negative_scores=negative_scores_typeof,
                label_smoothing=label_smoothing,
                batch_filter=batch_filter,
                num_entities=num_entities,
            ) * margin_typeof
        else:
            loss_typeof = torch.tensor(0.0)

        return loss_typeof + loss_subclassof + loss_rel


class TransKLTrainingLoop(SLCWATrainingLoop):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    
    @staticmethod
    def _process_batch_static(
        model: Model,
        loss: Loss,
        mode: Optional[InductiveMode],
        batch,
        start: Optional[int],
        stop: Optional[int],
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # get current time

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        relation_label = model.triples_factory.relation_to_id

        # filtra le relazioni typeof da positive_batch
        typeof_rel = positive_batch[:, 1] == relation_label["typeof"]
        subclass_rel = positive_batch[:, 1] == relation_label["subclassof"]
        positive_rel = ~(typeof_rel | subclass_rel)

        rels_typeof_positive = positive_batch[typeof_rel]
        rels_subclassof_positive = positive_batch[subclass_rel]
        positive_batch = positive_batch[positive_rel]

        if len(rels_typeof_positive) > 0:
            positive_scores_typeof = model.score_hrt(rels_typeof_positive, mode=mode, type="typeof")
        else:
            positive_scores_typeof = torch.tensor([])
        
        if len(rels_subclassof_positive) > 0:
            positive_scores_subclassof = model.score_hrt(rels_subclassof_positive, mode=mode, type="subclass")
        else:
            positive_scores_subclassof = torch.tensor([])

        if len(positive_batch) > 0:
            positive_scores = model.score_hrt(positive_batch, mode=mode, type="rel")
        else:
            positive_scores = torch.tensor([])

        typeof_negative_rel = negative_batch[:, 1] == relation_label["typeof"]
        subclass_negative_rel = negative_batch[:, 1] == relation_label["subclassof"]
        rels_typeof_negative = negative_batch[typeof_negative_rel]
        rels_subclassof_negative = negative_batch[subclass_negative_rel]
        negative_batch = negative_batch[~(typeof_negative_rel | subclass_negative_rel)]

        if len(rels_typeof_negative) > 0:
            negative_scores_typeof = model.score_hrt(rels_typeof_negative, mode=mode, type="typeof")
        else:
            negative_scores_typeof = torch.tensor([])

        if len(rels_subclassof_negative) > 0:
            negative_scores_subclassof = model.score_hrt(rels_subclassof_negative, mode=mode, type="subclass")
        else:
            negative_scores_subclassof = torch.tensor([])

        if len(negative_batch) > 0:
            negative_scores = model.score_hrt(negative_batch, mode=mode, type ="rel")
        else:
            negative_scores = torch.tensor([])
        
        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                positive_scores_typeof=positive_scores_typeof,
                negative_scores_typeof=negative_scores_typeof,
                positive_scores_subclassof=positive_scores_subclassof,
                negative_scores_subclassof=negative_scores_subclassof,
                margin = model.margin,
                margin_typeof = model.margin_typeof,
                margin_subclassof = model.margin_subclassof,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=model._get_entity_len(mode=mode),
            )
            + model.collect_regularization_term()
        )
            

is_grid_search = False
if is_grid_search:
    search = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    i = 0
    j = 2
    k = 1
    start = True
    # assegna le grigle di ricerca ai 3 diversi iperparametri
    while i < len(search):
        if not start:
            j = 0
        while j < len(search):
            if not start:
                k = 0
            while k < len(search):
                start = False
                try:
                    pipeline_result = pipeline(
                    training='D:/TransC-OWL/data/'+str(dataset)+'/Train/train2id.tsv',
                    testing='D:/TransC-OWL/data/'+str(dataset)+'/Test/test2id.tsv',
                    model=TransKL,
                    model_kwargs=dict(margin=search[i], margin_typeof=search[j], margin_subclassof=search[k], is_grid_search=True),
                    training_loop=TransKLTrainingLoop,
                    regularizer=NoRegularizer,
                    negative_sampler=TransKLNegativeSampler,
                    loss = TransKLLoss,
                    training_kwargs=dict(num_epochs=20, batch_size=256, callbacks=[BatchSaveModel], checkpoint_name='modelDBPEDIA15K-grid_search'+str(i)+str(j)+str(k)+'.pt', checkpoint_frequency=0),
                    device='gpu')
                except:
                    k += 1
                    continue
                k += 1
                    
            j += 1
        i += 1

else:
    pipeline_result = pipeline(
    training='D:/TransC-OWL/data/'+str(dataset)+'/Train/train2id.tsv',
    testing='D:/TransC-OWL/data/'+str(dataset)+'/Test/test2id.tsv',
    model=TransKL,
    model_kwargs=dict(margin=0.001, margin_typeof=10, margin_subclassof=1, is_grid_search=False, embedding_dim=embedding_dim),
    training_loop=TransKLTrainingLoop,
    regularizer=NoRegularizer,
    negative_sampler=TransKLNegativeSampler,
    loss = TransKLLoss,
    training_kwargs=dict(num_epochs=10000, batch_size=512, callbacks=[BatchSaveModel], checkpoint_name='model'+dataset+"-39.pt", checkpoint_frequency=5),
    device='gpu',
)
