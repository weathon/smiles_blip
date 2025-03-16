from transformers import PreTrainedModel, BlipTextModel, BlipProcessor, BlipVisionModel, AutoConfig, BlipForConditionalGeneration
from PIL import Image
import torch
import requests

class ITMModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.text_model = self.blip.text_decoder
        self.vision_encoder = self.blip.vision_model
        self.lm_loss = torch.nn.CrossEntropyLoss()
        self.itc_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, mode="lm"):
        if mode == "lm":
            return self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        elif mode == "itc":
            text_outputs = self.text_model(input_ids=input_ids, is_decoding=False)
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            vision_embeds = vision_outputs.last_hidden_state[:, 0, :]
            cosine_sim = torch.nn.functional.cosine_similarity(text_embeds, vision_embeds)
            return torch.sigmoid(cosine_sim) 
        else:
            raise ValueError("Invalid mode. Choose either 'lm' or 'itc'.")


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")
    model = ITMModel(config)

    # test lm
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    text = "An image of a dog"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(raw_image, text, return_tensors="pt")
    outputs = model(**inputs, mode="lm")
    
    # test itc
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    text = "An image of a cat"
    inputs = processor(raw_image, text, return_tensors="pt")
    outputs = model(**inputs, mode="itc")
    print(outputs)