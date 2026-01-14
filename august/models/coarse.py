from torch import nn, Tensor
import torch

class CoarseClassifier(nn.Module):
    def __init__(self, label_dict=None, device=None):
        super(CoarseClassifier, self).__init__()
        self.adapter_coarse_branch = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2)
        )
        self.device = device
        # coarse-classifier
        self.location_classifier = nn.Linear(1024, 5)  # 5 ['antrum', 'body', 'cardia', 'fundus', 'prepylorus']
        self.helicobacter_classifier = nn.Linear(1024, 2)  # ['negative', 'positive']
        self.category_classifier = nn.Linear(1024, 4)
        self.label_dict = label_dict

    def forward(self, slide_features, target_dict):
        loss = 0.0
        tasks = {
            "location": self.location_classifier,
            "helicobacter": self.helicobacter_classifier,
            "category": self.category_classifier,
        }
        slide_features = self.adapter_coarse_branch(slide_features)
        for key, classifier in tasks.items():
            if key in target_dict and target_dict[key] is not None:
                labels = torch.tensor(target_dict[key], device=self.device)
                logits = classifier(slide_features)
                loss += F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def inference(self, question_type, slide_features):
        """
        Coarse inference by averaging features from multiple image crops.

        Args:
            question_type (str): The input question string.
            slide_features (torch.Tensor): Image features from multiple crops [N, feature_size].
        Returns:
            predicted_answer (str): The generated answer string.
        """
        slide_features = self.adapter_coarse_branch(slide_features)  # [N, hidden_size]
        # Average features across crops
        if question_type == 'location':
            logits = self.location_classifier(slide_features)  # [N, num_location_classes]
            predicted_class = torch.argmax(logits).item()
            predicted_answer = self.label_dict['location'][predicted_class]
            if predicted_class == 0 or predicted_class == 4:
                return f"This slide represents the distal stomach, specifically the {predicted_answer}."
            return f"This slide represents the proximal stomach, specifically the {predicted_answer}."
        if question_type == 'helicobacter':
            logits = self.helicobacter_classifier(slide_features)  # [N, num_helicobacter_classes]
            predicted_class = torch.argmax(logits).item()
            predicted_answer = self.label_dict['helicobacter'][predicted_class]
            return f"This slide is {predicted_answer} for Helicobacter pylori infection."
        if question_type == 'category':
            logits = self.category_classifier(slide_features)
            predicted_class = torch.argmax(logits).item()
            predicted_answer = self.label_dict['category'][predicted_class]
            return f"The category of diagnosis represented in this slide is {predicted_answer}."
        assert False, "Unknown question type"
