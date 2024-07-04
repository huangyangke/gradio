import gradio as gr
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 文字类输入输出
# def greet(name):
#     return "hello " + name + "!"

# iface = gr.Interface(fn=greet, inputs=gr.Textbox(lines=5,placeholder="name here",label='name:'), 
#                      outputs=gr.Textbox(label='Greeting'))

# 图像类输入输出
# def turn_grey(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     return gray

# iface = gr.Interface(fn=turn_grey, inputs=gr.Image(), outputs = 'image')

# 音频类输入输出
# def file_path(input):
#     return input

# iface = gr.Interface(fn=file_path, inputs=gr.Audio(sources=["microphone"],type='filepath'), outputs = 'text')

# 常见输入组件
# input_list = [
#     gr.Audio(sources=['microphone','upload'],type='numpy',label='Audio File'),
#     gr.Checkbox(label='Checkbox'),
#     gr.ColorPicker(label='Color Picker'),
#     gr.DataFrame(label='Dataframe'),
#     gr.Dropdown(['option 1','option 2','option 3'], label='Dropdown'),
#     gr.File(label='File',type='filepath'),
#     gr.Image(sources=['webcam','upload'], label='Image'),
#     gr.Number(label='Number'),
#     gr.Radio(['option 1','option 2','option 3'], label='Radio'),
#     gr.Slider(minimum=0, maximum=10, label='Slider', step=5),
#     gr.Textbox(label='Textbox',lines=3,max_lines=7,placeholder='Placeholder'),
#     gr.TextArea(label='TextArea',lines=3,max_lines=7,placeholder='Placeholder'),
#     gr.Video(sources=['webcam','upload'],label='Video'),
#     gr.CheckboxGroup(['option 1','option 2','option 3'],label='Checkbox Group')
# ]

# output_list = [
#     gr.Textbox(label='Audio outputs'),
#     gr.Textbox(label='Checkbox outputs'),
#     gr.Textbox(label='Color Picker outputs'),
#     gr.Textbox(label='Dataframe outputs'),
#     gr.Textbox(label='Dropdown outputs'),
#     gr.Textbox(label='File outputs'),
#     gr.Textbox(label='Image outputs'),
#     gr.Textbox(label='Number outputs'),
#     gr.Textbox(label='Radio outputs'),
#     gr.Textbox(label='Slider outputs'),
#     gr.Textbox(label='Textbox outputs'),
#     gr.Textbox(label='TextArea outputs'),
#     gr.Textbox(label='Video outputs'),
#     gr.Textbox(label='Checkbox Group outputs'),
# ]

# def input_and_output(*input_data):
#     return input_data

# iface = gr.Interface(fn=input_and_output,
#                      inputs=input_list,
#                      outputs=output_list,
#                      title="Input and Output",
#                      description='This is a test of all the input types.',
#                      live=False)

# def audio_fn(audio):
#     hz = audio(0)
#     data = audio(1)
#     return hz, data

# iface = gr.Interface(fn=audio_fn, inputs=gr.Audio(type="numpy"), outputs="audio")
# iface.launch()

# simple = pd.DataFrame(
#     {
#         "a":[1,2,3],
#         "b":[4,5,6]
#     }
# )

# iface = gr.Interface(fn=None, inputs=None, outputs=gr.BarPlot(simple, x='a', y='b'))
# iface.launch()

# def process():
#     imageurl_list = [
#         "https://gips2.baidu.com/it/u=1651586290,17201034&fm=3028&app=3028&f=JPEG&fmt=auto&q=100&size=f600_800"
#     ]
#     images = [(image, f'image {i+1}') for i, image in enumerate(imageurl_list)]
#     return images

# iface = gr.Interface(fn=process,inputs=None, outputs=gr.Gallery(columns=5))
# iface.launch()

# def process():
#     Fs = 8000
#     f = 5
#     sample = 10
#     x = np.arange(sample)
#     y = np.sin(2 * np.pi * f * x / Fs)
#     #plt.plot(x,y)
#     plt.bar(x, y)
#     return plt

# iface = gr.Interface(fn=process,inputs=None, outputs=gr.Plot())
# iface.launch()


# def fig_output():
#     return "hello world!"

# iface = gr.Interface(fn=fig_output, inputs=None, outputs = gr.Textbox())
# iface.launch()

# def fig_output():
#     json_sample = {'name': 'John', 'age':30, 'city':"New York"}
#     return json_sample
# iface = gr.Interface(fn=fig_output, inputs=None, outputs = gr.Json())
# iface.launch()

# iface = gr.Interface(fn=None, inputs=None, outputs = gr.HTML(value='<h1>hello</h1>'))
# iface.launch()

# gr.Blocks()
# gr.Row()
# gr.Column()
# gr.Tab()
# gr.Group()
# gr.Accordion()

with gr.Blocks() as demo:
    with gr.Tab(label="txt2img"):
        with gr.Row():
            with gr.Column(scale=6):
                txt1 = gr.Textbox(lines=2, label="")
                txt2 = gr.Textbox(lines=2, label="")
            with gr.Column(scale=1,min_width=3):
                button1 = gr.Textbox(value="1")
                button2 = gr.Textbox(value="2")   
                button3 = gr.Textbox(value="3")           
                button4 = gr.Textbox(value="4")   
            with gr.Column(scale=1):
                generate_button = gr.Button(value='Generate',variant='primary',scale=1)
                with gr.Row():
                    dropdown = gr.Dropdown(['1','2','3','4'], label='Style1')
                    dropdown2 = gr.Dropdown(['1','2','3','4'], label='Style2')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dropdown3 = gr.Dropdown(['1','2','3','4'], label='Sampling method')
                    slider1 = gr.Slider(minimum=0, maximum=100, label='Sampling steps')
                checkboxgroup = gr.CheckboxGroup(['Resotre faces', 'Tiling', "Hires.fix"])   
                with gr.Row():
                    slider2 = gr.Slider(minimum=0,maximum=100,label='Width')
                    slider3 = gr.Slider(minimum=0,maximum=100,label='Batch count')
                with gr.Row():
                    slider4 = gr.Slider(minimum=0,maximum=100,label='Height')
                    slider5 = gr.Slider(minimum=0, maximum=100, label='Batch size')
                slider6 = gr.Slider(minimum=0, maximum=100, label='CFG scale')
                with gr.Row():
                    number1 = gr.Number(label='seed')
                    button5 = gr.Button(value='Randomize')
                    button6 = gr.Button(value='Reset')
                    checkbox1 = gr.Checkbox(label="Extra")
                dropdown4 = gr.Dropdown(['1','2','3','4'], label='Script')
        
demo.launch()