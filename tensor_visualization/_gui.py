'''
only visualize() is exported
'''

import sys 

from vispy import scene
from vispy.color import get_colormap, ColorArray
from vispy.visuals import transforms

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QSpinBox
from PyQt5.QtWidgets import QTreeWidgetItem, QTreeWidget
from PyQt5.QtWidgets import QWidget, QFrame
from PyQt5.QtWidgets import QBoxLayout, QHBoxLayout, QGridLayout

from PyQt5.QtCore import Qt

import numpy as np
import numpy.random as rd

from ._tensor_rank import TensorData


class ScatterRankShowerUnit(scene.visuals.Markers):
    def __init__(self, rank, orig_size):
        ...


def get_shower_unit(rank, orig_size, rank_viewmode):
    if rank_viewmode == 'hybrid':
        return ScatterRankShowerUnit(rank, orig_size)
    else:
        raise ValueError("{} not implemented".format(rank_viewmode))

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


class TensorDisplayContext(object):
    def __init__(self, tensor_data):
        self.canvas = CostumizedCanvas(keys='interactive')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.canvas.central_widget.remove_widget(self.view)
        self.canvas.central_widget.add_widget(self.view)
        tensor_data.register(self)

        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        self.axis.transform = transforms.MatrixTransform()
        self.axis.transform.translate((-1, -1, -1))
        self.shower_units = []

        self.rank_viewmode = "hybrid"

    def _clear_all_shower_units(self):
        for unit in self.shower_units:
            unit.parent = None
        self.shower_units = []
    
    def on_tensor_data_update(self, tensor_data):
        self._clear_all_shower_units()
        for l in range(tensor_data.partitions[0]):
            for m in range(tensor_data.partitions[1]):
                for n in range(tensor_data.partitions[2]):
                    shower_unit = get_shower_unit(
                              tensor_data.rank(l, m, n)
                            , (5, 5, 5)
                            , self.rank_viewmode
                            )
                    self.shower_units.append(shower_unit)
                    self.view.add(shower_unit)


class TensorRankViewSelector(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QFrame {border:1px solid;}')

        layout = QHBoxLayout(self)

        self.point_cloud = QRadioButton('Point Cloud')
        layout.addWidget(self.point_cloud)

        self.pure_color = QRadioButton('Pure Color')
        layout.addWidget(self.pure_color)

        self.hybrid = QRadioButton('Hybrid')
        self.hybrid.setChecked(True)
        layout.addWidget(self.hybrid)

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

class Context(object):
    def __init__(self):
        self.qt_app = QApplication([])
        self.tensor_data = TansorData()
        self.win = Window()
        

_context = Context()


def visualize(tensor):
    _context.tensor_data.reset()
    _context.tensor_data.setTensor(tensor)
    _context.tensor_data.update()

    _context.qt_app.exce_()
