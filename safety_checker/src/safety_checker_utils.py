import torch
from torch import nn

# Reference:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def detect_bad_concepts(model, clip_input):
    pooled_output = model.vision_model(clip_input)[1]  # pooled_output
    image_embeds = model.visual_projection(pooled_output)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    special_cos_dist = (
        cosine_distance(image_embeds, model.special_care_embeds).cpu().float().numpy()
    )
    cos_dist = cosine_distance(image_embeds, model.concept_embeds).cpu().float().numpy()

    result = []
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {},
            "special_care": [],
            "concept_scores": {},
            "bad_concepts": [],
        }

        # increase this value to create a stronger `nfsw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        for concept_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concept_idx]
            concept_threshold = model.special_care_embeds_weights[concept_idx].item()
            result_img["special_scores"][concept_idx] = round(
                concept_cos - concept_threshold + adjustment,
                3,
            )
            if result_img["special_scores"][concept_idx] > 0:
                result_img["special_care"].append(
                    {concept_idx, result_img["special_scores"][concept_idx]},
                )
                adjustment = 0.01

        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concept_idx]
            concept_threshold = model.concept_embeds_weights[concept_idx].item()
            result_img["concept_scores"][concept_idx] = round(
                concept_cos - concept_threshold + adjustment,
                3,
            )
            if result_img["concept_scores"][concept_idx] > 0:
                result_img["bad_concepts"].append(concept_idx)

        result.append(result_img)

    bad_concepts = [res["bad_concepts"] for res in result]

    bad_concepts_scores = [
        torch.asarray(list(res["concept_scores"].values())) for res in result
    ]
    bad_concepts_scores = torch.concat(bad_concepts_scores, dim=0)

    return bad_concepts, bad_concepts_scores
