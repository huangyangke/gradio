import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab('图生图'):
        gr.Markdown('# 图生图演示')
        with gr.Row():
            input_img = gr.Image(sources=['upload'], label='上传图片', type='pil')
            output_img = gr.Label
        # gr.Examples(['./dog.jpg'])