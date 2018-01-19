import sys 

from vispy import scene
from vispy.color import get_colormap, ColorArray
from vispy.visuals import transforms

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QSpinBox, QMessageBox
from PyQt5.QtWidgets import QTreeWidgetItem, QTreeWidget
from PyQt5.QtWidgets import QWidget, QFrame
from PyQt5.QtWidgets import QBoxLayout, QHBoxLayout, QGridLayout

from PyQt5.QtCore import Qt

import numpy as np
import numpy.random as rd

class CostumizedCanvas(scene.SceneCanvas):
    def __init__(self, *args, **kv):
        super().__init__(*args, **kv)
        self.unfreeze()

    def on_close(self, event):
        try:
            self.on_close_handler(self, event)
        except AttributeError:
            pass

    def set_on_close_handler(self, handler):
        self.on_close_handler = handler

class Context:
    def __init__(self):
        #self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        #self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        #self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas = CostumizedCanvas(keys='interactive')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.canvas.central_widget.remove_widget(self.view)
        self.canvas.central_widget.add_widget(self.view)
    
    def show(self):
        self.canvas.show()

_context = Context()

class TensorRankViewSelector(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QFrame {border:1px solid;}')

        layout = QHBoxLayout(self)
        #self.setLayout(layout)

        self.point_cloud = QRadioButton('Point Cloud')
        self.point_cloud.toggled.connect(self.point_cloud_handler)
        layout.addWidget(self.point_cloud)

        self.pure_color = QRadioButton('Pure Color')
        self.pure_color.toggled.connect(self.pure_color_handler)
        layout.addWidget(self.pure_color)

        self.hybrid = QRadioButton('Hybrid')
        self.hybrid.setChecked(True)
        layout.addWidget(self.hybrid)

        self.inner_hybrid_handler = None 
        self.inner_pure_color_handler = None 
        self.inner_point_cloud_handler = None

        self.hybrid.toggled.connect(self.hybrid_handler)
        self.current_view = "hybrid"

        self.observers = []

    def register(self, a_observer):
        self.observers.append(a_observer)

    def unregister(self, a_observer):
        self.observers.remove(a_observer)

    def update(self):
        for ob in self.observers:
            ob.on_tensor_rank_view_change(self.current_view)

    def hybrid_handler(self, enabled):
        if not enabled: return
        
        if not self.inner_hybrid_handler:
            QMessageBox.information(None, "Changement de Mode", "Bonjour!")
        else:
            self.inner_hybrid_handler()
            self.update()

    def point_cloud_handler(self, enabled):
        if not enabled: return

        if not self.inner_point_cloud_handler:
            QMessageBox.warning(None, "Changement de Mode", "Bonjour Nuage des Points")
        else:
            self.inner_point_cloud_handler()
            self.update()

    def pure_color_handler(self, enabled):
        if not enabled: return

        if not self.inner_pure_color_handler:
            QMessageBox.critical(None, "Changement de Mode", "Bonjour Couleur")
        else:
            self.inner_pure_color_handler()
            self.update()

    def add_switch_to_point_cloud(self, handler):
        self.inner_point_cloud_handler = handler

    def add_switch_to_pure_color(self, handler):
        self.inner_pure_color_handler = handler

    def add_switch_to_hybrid(self, handler):
        self.inner_hybrid_handler = handler


class TensorRankShower(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QFrame {border:1px solid;}')

        data_len_per_dim = 5
        layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        label_container = QWidget(self)
        layout.addWidget(label_container)
        label_container_layout = QHBoxLayout(label_container)
        label = QLabel('Tensor Rank', self)
        label.setAlignment(Qt.AlignHCenter)
        label_container_layout.addWidget(label)
        show_all_button = QPushButton('Expand/Collapse', self)
        label_container_layout.addWidget(show_all_button)

        tw = QTreeWidget(self)
        layout.addWidget(tw)
        tw.setColumnCount(2)
        tw.setHeaderLabels(["where", "rank", "size"])

        for x in range(data_len_per_dim):
            x_child = QTreeWidgetItem(["x:{}".format(x)])
            for y in range(data_len_per_dim):
                y_child = QTreeWidgetItem(["y:{}".format(y)])
                for z in range(data_len_per_dim):
                    z_child = QTreeWidgetItem(["z:{}".format(z), "({},{},{})".format(1, 1, 1), "{}x{}x{}".format(10, 10, 10)])
                    y_child.addChild(z_child)
                x_child.addChild(y_child)
            tw.addTopLevelItem(x_child)

        self.is_expanded = False
        def toggle_expand():
            tw.collapseAll() if self.is_expanded else tw.expandAll()
            self.is_expanded = not self.is_expanded

        show_all_button.clicked.connect(toggle_expand)

    def on_tensor_rank_view_change(self, new_view):
        ...

class XYZRangeSelector(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QFrame {border:1px solid;}')

        layout = QGridLayout(self)

        modes = ['X', 'Y', 'Z']
        for m in range(1, 4):
            mode = modes[m - 1]

            label = QLabel(mode, self)
            label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(label, 1, m)

            how_many_partition_box = QSpinBox(self)
            how_many_partition_box.setToolTip("how many partition?")
            layout.addWidget(how_many_partition_box, 2, m)
        
            start_box = QSpinBox(self)
            start_box.setToolTip("start from")
            layout.addWidget(start_box, 3, m)

            end_box = QSpinBox(self)
            end_box.setToolTip("end at")
            layout.addWidget(end_box, 4, m)

        rerender_button = QPushButton('Rerender')
        layout.addWidget(rerender_button, 5, 3)
            
class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.resize(1200, 800)
        self.setWindowTitle('Tensor Rank Visualization')

        def canvas_on_close_handler(_1, _2):
            self.close()
        _context.canvas.set_on_close_handler(canvas_on_close_handler)
        box.addWidget(_context.canvas.native)
        
        rightBoxWidget = QWidget()
        rightBox = QBoxLayout(QBoxLayout.TopToBottom, rightBoxWidget)
        rightBox.setAlignment(Qt.AlignTop)
        box.addWidget(rightBoxWidget)

        tensor_view_selector = TensorRankViewSelector()
        rightBox.addWidget(tensor_view_selector)

        xyzRangeSelector = XYZRangeSelector()
        rightBox.addWidget(xyzRangeSelector)

        tw_container = TensorRankShower()
        rightBox.addWidget(tw_container)

        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

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

def add_scatter_one_by_one():
    how_many_seg = np.array((5, 5, 5))
    to_scale = 0.4
    for l in range(how_many_seg[0]):
        for m in range(how_many_seg[1]):
            for n in range(how_many_seg[2]):
                scatter = scene.visuals.Markers()
                poss = np.random.random(size=(10000, 3)) if l == m and l == n else np.random.random(size=(10, 3))
                poss += (l, m, n)
                poss -= how_many_seg/2
                poss *= to_scale

                scatter.set_data(poss, face_color=(1, 0, 0, 0.5) if l == m and l == n else (1, 1, 1, 0.5), size=5)
                _context.view.add(scatter)
                #_context.view.update()
    axis = scene.visuals.XYZAxis(parent=_context.view.scene)
    axis.transform = transforms.MatrixTransform()
    axis.transform.translate((-1, -1, -1))


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
    #add_scatter()
    add_scatter_one_by_one()
    qt_app = QApplication(sys.argv)
    ex = Window()
    qt_app.exec_()

    #add_cubes()
    #_context.canvas.app.run()
