import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random
import numpy as np
import torch
from io import BytesIO
from detections.ultis import show, show_everything, model_predict_click, model_predict_box, model_predict_everything


def click(container_width, ground, height, scale, radius_width, show_mask, model, im):
    """
    """
    canvas_result = st_canvas(
            fill_color="rgba(11, 240, 11, 0.3)" if ground == "Foreground" else "rgba(173, 50, 173, 0.8)",
            background_image = st.session_state['im'],
            drawing_mode='point',
            width = container_width,
            height = height * scale,
            point_display_radius = radius_width,
            stroke_width=2,
            update_streamlit=True,
            key="click",)
    
    if not show_mask:
        im = Image.fromarray(im).convert("RGB")
        
        if im != st.session_state['im']:       
            st.session_state['im'] = im

    elif canvas_result.json_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])

        if len(df) == 0:
            st.session_state.clear()
            if "canvas_result" not in st.session_state:
                st.session_state["canvas_result"] = len(df)
            
            elif len(df) != st.session_state["canvas_result"]:
                st.session_state["canvas_result"] = len(df)

        df["center_x"] = df["left"]
        df["center_y"] = df["top"]

        input_points = []
        input_labels = []

        for _, row in df.iterrows():
            x, y = row["center_x"] + 5, row["center_y"]
            x = int(x/scale)
            y = int(y/scale)
            input_points.append([x,y])
            if row["fill"] == "rgba(11, 240, 11, 0.3)":
                input_labels.append(1) # foreground
            else:
                input_labels.append(0) # background
        
        # predict masks
        masks = model_predict_click(im, input_points, input_labels, model)

        #random color
        color = np.concatenate([np.random.randint(0,255, size=(3)), np.array([0.6])],axis=0) 
        
        if "input_masks_color" not in st.session_state:
            st.session_state["input_masks_color"] = [[masks, color]]

        elif not np.array_equal(st.session_state['input_masks_color'][-1][0], masks):
                st.session_state['input_masks_color'][-1] = [masks,color]
        
        im_masks = show(st.session_state["input_masks_color"])
        im_masks = Image.fromarray(im_masks).convert("RGBA")

        im = Image.alpha_composite(Image.fromarray(im).convert("RGBA"), im_masks).convert("RGB")
        torch.cuda.empty_cache()

        if im != st.session_state['im']:       
            st.session_state['im'] = im
        im_bytes = BytesIO()
        st.session_state['im'].save(im_bytes,format='PNG')
        st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')


def box(container_width,height,scale,radius_width,show_mask,model,im):
    for each in ['color_change_point','input_masks_color']:
        if each in st.session_state:st.session_state.pop(each)
    
    canvas_result_1 = st_canvas(
            fill_color="rgba(255, 255, 0, 0)",
            background_image = st.session_state['im'],
            drawing_mode='rect',
            stroke_color = "rgba(0, 255, 0, 0.6)",
            stroke_width = radius_width,
            width = container_width,
            height = height * scale,
            point_display_radius = 12,
            update_streamlit=True,
            key="box",
            )
    if not show_mask:
        im = Image.fromarray(im).convert("RGB")
        if im != st.session_state['im']:
            st.session_state['im'] = im

    elif canvas_result_1.json_data is not None:
        st.button('Save color')
        df = pd.json_normalize(canvas_result_1.json_data["objects"])
        if len(df) == 0:
            st.session_state.clear()
        if 'canvas_result' not in st.session_state:
            st.session_state['canvas_result'] = len(df)

        elif len(df) != st.session_state['canvas_result']:
            st.session_state['canvas_result'] = len(df)


        center_point,center_label,input_box = [],[],[]
        for _, row in df.iterrows():
            x, y, w,h = row["left"], row["top"], row["width"], row["height"]
            x = int(x/scale)
            y = int(y/scale)
            w = int(w/scale)
            h = int(h/scale)
            center_point.append([x+w/2,y+h/2])
            center_label.append([1])
            input_box.append([x,y,x+w,y+h])
        

        masks = model_predict_box(im,center_point,center_label,input_box,model)
        masks = np.array(masks)
        
        # random color
        color = np.concatenate([np.random.randint(0,255, size=(3)), np.array([0.6])],axis=0)

        if 'input_masks_color_box' not in st.session_state:
            st.session_state['input_masks_color_box'] = [[masks,color]]
            
        elif not np.array_equal(st.session_state['input_masks_color_box'][-1][0],masks):
            st.session_state['input_masks_color_box'][-1] = [masks,color]
        im_masked = show(st.session_state['input_masks_color_box'])
        im_masked = Image.fromarray(im_masked).convert('RGBA')
        im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
        torch.cuda.empty_cache()

        if im != st.session_state['im']:
            st.session_state['im'] = im

        im_bytes = BytesIO()
        st.session_state['im'].save(im_bytes,format='PNG')
        st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')


def everthing(im,show_mask,model):
    st.session_state.clear()
    everything = st.image(Image.fromarray(im))
    if show_mask:
        masks = model_predict_everything(im,model)
        im_masked = show_everything(masks)
        im_masked = Image.fromarray(im_masked).convert('RGBA')
        im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
        everything.image(im)
        torch.cuda.empty_cache()
        im_bytes = BytesIO()
        im.save(im_bytes,format='PNG')
        st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')        



def main():
    print('init')
    torch.cuda.empty_cache()
    with st.sidebar:
        im = st.file_uploader(label='Upload image',type=['png','jpg','tif'])
        option = st.selectbox(
            'Segmentation mode',
            ('Click', 'Box', 'Everything'))
        model = st.selectbox(
            'Model',
            ('vit_b', 'vit_l', 'vit_h'))
        ground = st.sidebar.radio("Ground", ["Foreground", "Background"])
        show_mask = st.checkbox('Show mask',value = True)
        radius_width = st.slider('Radius/Width for Click/Box',0,20,5,1)
        
    if im:
        im = Image.open(im).convert("RGB")
        if 'im' not in st.session_state:
            st.session_state['im'] = im
        width, height   = im.size[:2]
        im              = np.array(im)
        container_width = 700
        scale           = container_width/width
        if option == 'Click':
            click(container_width, ground, height,scale,radius_width,show_mask,model,im)
        if option == 'Box':
            box(container_width,height,scale,radius_width,show_mask,model,im)
        if option == 'Everything':
            everthing(im,show_mask,model)
    else:
        st.session_state.clear()


if __name__ == "__main__":
    main()