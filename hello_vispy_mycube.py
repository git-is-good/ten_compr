import sys 

from vispy import scene
from vispy.color import get_colormap, ColorArray
from vispy.visuals import transforms

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton

import numpy as np

class Context:
    def __init__(self):
        #self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        #self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.canvas = scene.SceneCanvas(keys='interactive')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
    
    def show(self):
        self.canvas.show()

_context = Context()

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        box = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.resize(1200, 800)
        box.addWidget(_context.canvas.native)
        

        rightBoxWidget = QtWidgets.QWidget()
        rightBox = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, rightBoxWidget)
        box.addWidget(rightBoxWidget)

        button = QPushButton('Change View', self)
        button.setToolTip('Change View Port')
        rightBox.addWidget(button)

        button2 = QPushButton('Change Model', self)
        button2.setToolTip('Change Model of the Tensor')
        rightBox.addWidget(button2)

        self.show()

def add_scatter():
    how_many_seg = 5
    to_scale = 0.4
    poss = []
    for l in range(how_many_seg):
        for m in range(how_many_seg):
            for n in range(how_many_seg):
                if l == m and l == n:
                    for _ in range(10000):
                        poss.append((
                              (np.random.random() + l - how_many_seg/2) * to_scale
                            , (np.random.random() + m - how_many_seg/2) * to_scale
                            , (np.random.random() + n - how_many_seg/2) * to_scale
                        ))
                else:
                    for _ in range(10):
                        poss.append((
                              (np.random.random() + l - how_many_seg/2) * to_scale
                            , (np.random.random() + m - how_many_seg/2) * to_scale
                            , (np.random.random() + n - how_many_seg/2) * to_scale
                        ))

    poss = np.array(poss)
    scatter = scene.visuals.Markers()
    scatter.set_data(poss, edge_color=None, face_color=(1, 1, 1, 0.5), size=5)
    _context.view.add(scatter)


def add_cubes():
    cm = get_colormap('hot')
    how_many_seg = 5
    for l in range(how_many_seg):
        for m in range(how_many_seg):
            for n in range(how_many_seg):
                cube_size = 0.15
                how_much_red = np.random.random()
                cube = scene.visuals.Sphere(
                          radius=cube_size/2
                        , method='latitude'
                        , color=cm[np.random.random()]
                        )
                #cube = scene.visuals.Cube((cube_size, cube_size, cube_size),
                        #color=ColorArray((how_much_red, 0, 0, 0.8)))#,
                        #edge_color='k')
                cube.transform = transforms.MatrixTransform()
                to_translate = 0.4
                cube.transform.translate((to_translate*(l-how_many_seg//2), to_translate*(m-how_many_seg//2), to_translate*(n-how_many_seg//2)))
                _context.view.add(cube)


if __name__ == '__main__':
    add_scatter()
    qt_app = QtWidgets.QApplication([])
    ex = Window()
    qt_app.exec_()

    #add_cubes()
    _context.canvas.app.run()
