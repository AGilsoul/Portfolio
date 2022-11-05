import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from VectorPlot import create_plot, create_empty_plot
from VectorUtils import *


AppFont = 'Any 16'
sg.theme('LightGrey')

exit_layout = [
    [sg.Button('Exit', font=AppFont)]
]

vector_input_layout = [
    [sg.Text('V', key='V', size=(2, 1)),
     sg.InputText(key='V1', size=(2, 1), default_text='1'),
     sg.InputText(key='V2', size=(2, 1), default_text='0'),
     sg.InputText(key='V3', size=(2, 1), visible=False, default_text='0')],
    [sg.Text('U', key='U', size=(2, 1)),
     sg.InputText(key='U1', size=(2, 1), default_text='0'),
     sg.InputText(key='U2', size=(2, 1), default_text='1'),
     sg.InputText(key='U3', size=(2, 1), visible=False, default_text='0')],
    [sg.Button('Update', font=AppFont)]
]

button_layout = [
    [sg.Radio('2D', 'VectorDimension', key='2D', enable_events=True, default=True),
     sg.Checkbox('Show Projection V onto U', key='PROJ', default=False)],
    [sg.Radio('3D', 'VectorDimension', key='3D', enable_events=True, default=False),
     sg.Checkbox('Show Cross Product', key='CP', visible=False, default=False)]
]

overall_layout = [
    [sg.Canvas(key='figCanvas')],
    [
        sg.Column(vector_input_layout, element_justification='left', expand_x=True, size=(50,100)),
        sg.VSeparator(),
        sg.Column(button_layout, element_justification='left', expand_x=True, size=(100,50)),
        sg.VSeparator(),
        sg.Column(exit_layout, element_justification='right', expand_x=True)
     ]
]


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def update_figure(window, values):
    origins_dict = {}
    if values['2D']:
        V = [float(values['V' + str(i + 1)]) for i in range(0, 2)]
        U = [float(values['U' + str(i + 1)]) for i in range(0, 2)]
    else:
        V = [float(values['V' + str(i + 1)]) for i in range(0, 3)]
        U = [float(values['U' + str(i + 1)]) for i in range(0, 3)]

    vec_list = np.array([V, U])
    if values['CP']:
        cross_prod = Vector.cross_product(vec_list[0], vec_list[1])
        vec_list = np.concatenate([vec_list, [cross_prod]])
    if values['PROJ']:
        proj = Vector.vector_projection(vec_list[1], vec_list[0])
        vec_list = np.concatenate([vec_list, [proj[0], proj[1]]])
        origins_dict[len(vec_list) - 1] = proj[0]
    return draw_figure(window['figCanvas'].TKCanvas, create_plot(vec_list, origins_dict))


def main():
    window = sg.Window('Vector Plotting',
                       overall_layout,
                       finalize=True,
                       resizable=True)
    # fix this
    figure = update_figure(window, window.read(1)[1])
    while True:
        event, values = window.read()
        # print('event:')
        # print(event)
        # print('value:')
        # print(values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Update':
            figure.get_tk_widget().forget()
            plt.close('all')
            figure = update_figure(window, values)
        elif event == '2D':
            window['CP'].Update(visible=False)
            window['CP'].Update(False)
            window['V3'].Update(visible=False)
            window['U3'].Update(visible=False)
        elif event == '3D':
            window['CP'].Update(visible=True)
            window['V3'].Update(visible=True)
            window['U3'].Update(visible=True)
            window['V3'].Update('')
            window['U3'].Update('')


    window.close()


if __name__ == '__main__':
    main()