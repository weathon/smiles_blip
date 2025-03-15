# %%
bp = breakpoint
import random
from datasets import load_dataset
import numpy as np
from evaluate import load
import torch
from transformers import TrainingArguments, Trainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoProcessor
)
from charactertokenizer import CharacterTokenizer
import deepsmiles
converter = deepsmiles.Converter(rings=True, branches=True)

# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
from transformers import BlipForConditionalGeneration
checkpoint = "Salesforce/blip-image-captioning-base" 
# checkpoint = "weathon/smiles_llava"
config = AutoConfig.from_pretrained(checkpoint)
config.vision_config.dropout = 0.2
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", config=config)
processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[ITM]"]})
base_model = BlipForConditionalGeneration.from_pretrained(
    checkpoint,
)
# bp()
# base_model.resize_token_embeddings(len(processor.tokenizer)) this will cause error, but there are extract empty tokens already 


# %%
def encode_smiles(example):
    example["deepsmiles"] = converter.encode(example["caption"])
    return example
# ds = load_dataset("weathon/3d2smiles_real")
ds = load_dataset("weathon/3d2smiles_synthetic")

train_ds = ds["train"].map(encode_smiles)
test_ds = ds["val"].map(encode_smiles)
all_possible_chars = list(set("".join(train_ds["deepsmiles"])))

# %%
train_ds[0]["deepsmiles"]

# %%
# get cpu cores number
import os
os.cpu_count()

# %%


for param in base_model.parameters():
    param.requires_grad = False

for param in base_model.vision_model.parameters():
    param.requires_grad = True


# %% 
import torchvision.transforms as T

def transforms(examples):
  texts = [f"The SMILES of this molecule is {example}" for example in examples["deepsmiles"]]
  images = [example.resize((224, 224)) for example in examples["image"]]

  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

  batch["cid"] = examples["cid"]
  batch["deepsmiles"] = examples["deepsmiles"]
  labels = batch["input_ids"].clone()
  batch["labels"] = labels
  return batch

train_ds.set_transform(transforms)
test_ds.set_transform(transforms)

# %%
from transformers import PreTrainedModel
class Router(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config) 
        self.blip = base_model
        self.itm_head = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.itm_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, mode="lm"):
        if mode == "lm":
            outputs = self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            ids = torch.argmax(logits, dim=-1)
            loss = outputs.loss
            return {"loss": loss, "logits": logits, "ids": ids}

        elif mode == "itm":
            cls_token_id = processor.token_to_chars["[ITM]"]
            bp()
            cls_indices = (input_ids == cls_token_id).int().nonzero(as_tuple=True)[1]
            non_zero_cls = cls_indices[cls_indices!=0]
            end_index = non_zero_cls + 1
            print("end_index: ", end_index)
            bp()
            # needs to +1, because it needs to be what comes after the cls_token_id token, 
            # not at the cls_token_id token, but this means we need to add another eos at the input
            # it is cls, so the model knows it is a pred, not the same as end
            vision_outputs = self.blip.vision_model(
                pixel_values=pixel_values,
            )

            image_embeds = vision_outputs[0]

            outputs = self.blip.text_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                reduction="mean",
                output_hidden_states=True,
            )

            loss = outputs.loss
            last_hidden_state = outputs.hidden_states[-1]
            # print(last_hidden_state.shape)
            end_index = end_index.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 768)
            # print(end_index.shape)

            # print(torch.gather(last_hidden_state, 1, end_index).shape)
            logits = self.itm_head(torch.gather(last_hidden_state, 1, end_index)).squeeze(1)#[:, -1].unsqueeze(-1)
            # print("logits shape: ", logits.shape) 32x32, no wonder, shape not the same
            if labels is not None:
                # print(logits.shape, labels.shape)
                loss = self.itm_loss(logits, labels.float())
            else:
                loss = None
            return {"loss": loss, "logits": logits}
        else:
            raise ValueError("mode should be either 'lm' or 'itm'")


           	


model = Router(config)
inputs = train_ds[0:1]
# print(model(inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"], labels=inputs["labels"], mode="lm"))

# print(model(inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"], labels=torch.tensor([[1]]), mode="itm"))

# %%
error_bank = [] # a queue with max size 100 and a list of dict {"img": img, "pred": "wrong_smiles"}


# # init error bank with 100 samples
# for i in range(20):
#     while 1:
#         correct_index = torch.randint(0, len(train_ds), (1,))
#         correct_cid = train_ds[correct_index]["cid"]
#         correct_smiles = train_ds[correct_index]["deepsmiles"]
#         correct_image = train_ds[correct_index]["pixel_values"]
#         wrong_index = torch.randint(0, len(train_ds), (1,))
#         wrong_cid = train_ds[wrong_index]["cid"]
#         wrong_smiles = train_ds[wrong_index]["deepsmiles"]
#         if correct_cid != wrong_cid:
#             break
#     error_bank.append({
#         "img": correct_image,
#         "pred": wrong_smiles,
#     })
# do not use error bank, too few samples?
# %%
def get_itm_sample():
    label = torch.randint(0, 2, (1,))
    if label == 0:
        index = torch.randint(0, len(train_ds), (1,))
        img = train_ds[index]["pixel_values"]
        pred = train_ds[index]["deepsmiles"]
        print(pred)
        # replace randon k characters with random characters
        k = random.randint(1, 5)
        # print("Original SMILES: ", pred[0])
        tmp = list(pred[0])
        for i in range(k):
            index = random.randint(0, len(tmp) - 1)
            tmp[index] = random.choice(all_possible_chars)
        pred[0] = "".join(tmp)
        # print("Corrupted SMILES: ", pred[0])
    else:
        # matched, chose from train_ds
        index = torch.randint(0, len(train_ds), (1,))
        img = train_ds[index]["pixel_values"]
        pred = train_ds[index]["deepsmiles"]
    
    pred[0] += "ITM"
    # print("SMILES: ", pred[0])
    # breakpoint()

    return {
        "img": img,
        "text": pred,
        "label": label,
    }

# print(get_itm_sample())

def get_itm_batch(batch_size):
    pixel_values = []
    deepsmiles = []
    attention_mask = []
    labels = []
    for i in range(batch_size):
        sample = get_itm_sample()
        pixel_values.append(sample["img"][0])
        deepsmiles.append(sample["text"][0])
        labels.append(sample["label"])

    pixel_values = torch.stack(pixel_values)
    processed = processor.tokenizer(deepsmiles, padding=True, return_tensors="pt", return_attention_mask=True)
    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    labels = torch.stack(labels)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# print(get_itm_batch(2))
# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
training_config = {
    "lr": 5e-5,
    "batch_size": 32,
    "num_epochs": 30,
    "weight_decay": 0.02,
    "min_factor": 0.1,
}
import wandb
wandb.init(
    project="finetune",
    config=training_config,
)
train_dataloader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=os.cpu_count(),
)
test_dataloader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=training_config["batch_size"],
    shuffle=False,
    num_workers=os.cpu_count(),
)

device = "cuda"
model = model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config["lr"],
    weight_decay=training_config["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=training_config["num_epochs"] * len(train_dataloader),
    eta_min=training_config["lr"] * training_config["min_factor"],
)




def val():
    correct = 0
    total = 0     
    lm_losses = []
    itm_losses = []
    itm_acc = []
    print("Validation")
    for step, batch in enumerate(test_dataloader):
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                mode="lm",
            )
            loss = outputs["loss"]
            decoded_output = processor.batch_decode(outputs["ids"], skip_special_tokens=True)
            correct_output = batch["deepsmiles"]
            decoded_output = [x.replace(" ", "").replace("thesmilesofthismoleculeis", "") + "[ITM]" for x in decoded_output]
            correct_output = [x.replace(" ", "").lower() for x in correct_output]
            itm_gt = []
            for i in range(len(decoded_output)):
                if decoded_output[i] == correct_output[i]:
                    correct += 1
                    itm_gt.append(1)
                else:
                    itm_gt.append(0)
                total += 1
            lm_losses.append(loss.item())
            print(itm_gt)
            # use model generated smiles and math-or-not to get itm loss and acc
            itm_text = processor.tokenizer.batch_encode_plus(decoded_output, padding=True, return_tensors="pt")
            itm_outputs = model(
                input_ids=itm_text["input_ids"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                attention_mask=itm_text["attention_mask"].to(device),
                labels=torch.tensor(itm_gt).unsqueeze(1).to(device),
                mode="itm",
            )
            itm_loss = itm_outputs["loss"]
            itm_logits = itm_outputs["logits"]
            itm_pred = torch.sigmoid(itm_logits).cpu().detach().numpy() > 0.5
            itm_acc.extend((itm_pred == itm_gt).flatten())
            itm_losses.append(itm_loss.item())

            
    wandb.log({"val_lm_loss": sum(lm_losses) / len(lm_losses)})
    wandb.log({"val_acc": correct / total})
    wandb.log({"val_itm_loss": sum(itm_losses) / len(itm_losses)})
    wandb.log({"val_itm_acc": np.mean(itm_acc)})
            
    model.train()


# %%
for epoch in range(training_config["num_epochs"]):
    print(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
    model.train()
    for step, batch in enumerate(train_dataloader):
        # print("Training LM")
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            mode="lm",
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        wandb.log({"lm_loss": loss.item()})
        decoded_output = processor.batch_decode(outputs["ids"], skip_special_tokens=True)
        correct_output = batch["deepsmiles"]
        # print(decoded_output[0].replace(" ", "").replace("thesmilesofthismoleculeis", ""), correct_output[0].replace(" ", ""))
        decoded_output = [x.replace(" ", "").replace("thesmilesofthismoleculeis", "") for x in decoded_output]
        correct_output = [x.replace(" ", "").lower() for x in correct_output]
        
        # print("Training ITM")
        optimizer.zero_grad()
        itm_batch = get_itm_batch(32)
        outputs = model(
            input_ids=itm_batch["input_ids"].to(device),
            pixel_values=itm_batch["pixel_values"].to(device),
            attention_mask=itm_batch["attention_mask"].to(device),
            labels=itm_batch["labels"].to(device),
            mode="itm",
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        itm_logits = outputs["logits"]
        itm_pred = torch.sigmoid(itm_logits).cpu().detach().numpy() > 0.5
        itm_gt = itm_batch["labels"].cpu().numpy()
        itm_acc = (itm_pred == itm_gt).sum() / len(itm_gt)
        wandb.log({"itm_acc": itm_acc, "itm_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        scheduler.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            val()

        


