from gradio_client.serializing import ImgSerializable
import gradio as gr
import os
import torch
import numpy as np
import random
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import supervision as sv
from PIL import Image

HOME='/vol/src/gdino'

CONFIG_PATH = os.path.join(HOME, "GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"

WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model.eval()

transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def prompt_fn(text_prompt):
  labels = text_prompt.split('.')
  colors_ = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
  colors=colors_(len(labels))
  #color_map{={"white chicken": "#a89a00", "black chicken": "#a8ffea", "red chicken": "ffaeff"}
  color_map=dict(zip(labels,colors))
  return color_map

def gradio_predict(img,text_prompt,text_threshold,box_threshold):
  color_map=prompt_fn(text_prompt)
  tthreshold=text_threshold/100.0
  bthreshold=box_threshold/100.0
  im = Image.fromarray(img)
  image,_ = transform(im, None)

  bb=[]
  lo=[]
  ph=[]
  boxes, logits, phrases = predict(
    model=model,
    image=image,
    device='cuda:0',
    caption=text_prompt,
    box_threshold=bthreshold,
    text_threshold=tthreshold
  )

  for box,conf,label in zip(boxes, logits, phrases):
    bb.append(box.numpy())
    lo.append(conf.numpy())
    ph.append(label)
  bb = torch.tensor(bb)
  #frame = np.array(img, dtype="uint8")
  annotated_frame = annotate(image_source=img, boxes=bb, logits=lo, phrases=ph)
  return (annotated_frame)



with gr.Blocks() as demo:
    gr.Markdown("""
    **Grounding dino prompt predict ** Kobyzev Yuri.
    """)

    with gr.Row():
      with gr.Column():
        text_prompt=gr.Textbox(interactive=True,value='')
        text_threshold = gr.Slider(0, 100, value=35, interactive=True, label="text threshold", info="Choose between 0 and 100")
        box_threshold = gr.Slider(0, 100, value=45, interactive=True, label="boxth reshold", info="Choose between 0 and 100")

    with gr.Row():
        with gr.Column():
          img_input = gr.Image()
          predict_btn = gr.Button("Identify and segment Sections")

        with gr.Column():
          img_output = gr.Image()

    with gr.Row():

      clr_btn = gr.ClearButton([img_input,img_output],value = "Clear")


    predict_btn.click(gradio_predict, [img_input,text_prompt,text_threshold,box_threshold], img_output)

if __name__ == "__main__":
    demo.queue().launch(debug=True,share=True)
