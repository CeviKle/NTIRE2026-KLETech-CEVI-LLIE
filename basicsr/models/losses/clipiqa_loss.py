import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPIQA_Loss(nn.Module):
    def __init__(self,
                 model_name='ViT-B-32',
                 pretrained='openai',
                 loss_weight=1.0,
                 reduction='mean'):

        super(CLIPIQA_Loss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        # Quality prompt
        self.prompts = ["a high quality photo", "a low quality photo"]
        self.tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            text_tokens = self.tokenizer(self.prompts).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = F.normalize(self.text_features, dim=-1)

    def forward(self, x, gt=None):

        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = (x + 1) / 2  # normalize to [0,1] if input is [-1,1]

        image_features = self.model.encode_image(x)
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.text_features.T

        # We want image closer to "high quality"
        target = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, target)

        return loss * self.loss_weight