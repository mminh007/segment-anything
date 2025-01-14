import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
from io import BytesIO
from detections.ultis import show, model_predict_click, inpaint_model, sam_model

def sam_predict_mask(input_points, input_labels, im):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_points = [input_points]
    input_labels = [input_labels]

    sam, processor = sam_model()
    inputs = processor(im, input_points=input_points, input_labels=input_labels,
                       return_tensors=True).to(device)
    
    with torch.no_grad():
        outputs = sam(**inputs) # dict("iou_scores", "pred_masks")

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    ) # (1,1,3,512,512)

    scores = outputs.iou_scores # (1,1,3)

    return masks, scores


def main():
    torch.cuda.empty_cache()
    with st.sidebar:
        bg_image = st.file_uploader(label="Upload image", type=["png", "jpg"])
        
        ground = st.sidebar.radio("Ground", ["Foreground", "Background"])
        #show_mask = st.checkbox("Show mask", value = True)
        radius_width = st.slider("Radius/Width for Click/Box", 0,20,5,1)


    if bg_image is not None:
        im = Image.open(bg_image).convert("RGB")
    w, h = im.size[:2]
    scale_w, scale_h = w / 512, h / 512

    if "im" not in st.session_state:
        st.session_state["im"] = im
    
    #im = np.array(im)
        
    canvas_result = st_canvas(
            fill_color="rgba(11, 240, 11, 0.3)" if ground == "Foreground" else "rgba(173, 50, 173, 0.8)",
            background_image = st.session_state['im'] if bg_image else None,
            drawing_mode='point',
            stroke_color = "rgba(0, 255, 0, 0.6)",
            stroke_width = 2,
            width = 512,
            height = 512,
            point_display_radius = radius_width,
            update_streamlit=True,
            key="click",
    )

    left, right = st.columns(2)
    if canvas_result.json_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])

        if st.button("Predict Mask"):
            df["center_x"] = df["left"]
            df["center_y"] = df["top"]

            input_points = []
            input_labels = []

            for _, row in df.iterrows():
                x, y = row["center_x"], row["center_y"]
                x, y = x * scale_h, y * scale_w
                input_points.append([x,y])

                if row["fill"] == "rgba(11, 240, 11, 0.3)":
                    input_labels.append(1) # foreground
                else:
                    input_labels.append(0) # background
            
            
            masks, scores = sam_predict_mask(input_points, input_labels, im)
            if "input_mask_prediction" not in st.session_state:
                st.session_state["input_mask_prediction"] = [[masks[0], scores]]
            
            elif not np.array_equal(st.session_state['input_mask_prediction'][-1][0], masks[0]):
                st.session_state['input_mask_prediction'][-1] = [masks, scores]
            
            #-----------------------------------------------------------
        #     im_mask = show(st.session_state["input_mask_prediction"])
        #     if "im_mask" not in st.session_state:
        #         st.session_state["im_mask"] = im_mask

        #     elif not np.array_equal(st.session_state['im_mask'], im_mask):
        #         st.session_state["im_mask"] = im_mask
        
        #     # show mask
        #     with left:               
        #         im_mask = Image.fromarray(im_mask).convert("RGBA")
                
        #         st.header("Mask")
        #         st.image(im_mask)
        # #inpaint
        # prompt = st.text_input("Input prompt")
        # if st.button("Submit"):
        #     input_image = Image.fromarray(im)
        #     input_mask = Image.fromarray(st.session_state["im_mask"])

        #     input_image = input_image.resize((512,512))
        #     input_mask = input_mask.resize((512,512))

        #     pipe = inpaint_model()
        #     output = pipe(prompt = prompt, image = input_image,
        #                         mask_image = input_mask).images[0]
        #     #st.session_state["output_inpainting"].append([output])

        #     if "output_inpainting" not in st.session_state:
        #         st.session_state["output_inpainting"] = [output]

        #     elif not np.array_equal(st.session_state['output_inpainting'], output):
        #         st.session_state['output_inpainting'] = output

        #     # Show output
        #     with right:
        #         st.header("Output")
        #         st.image(st.session_state["output_inpainting"][-1])
            
        #     im_bytes = BytesIO()
        #     st.session_state["output_inpainting"].save(im_bytes, fomat="PNG")
        #     st.download_button("Download image", data=im_bytes.getvalue(), file_name="inpaint.png")
        #     torch.cuda.empty_cache()
    
        # st.write(df)