'''
only visualize() is exported
'''

import sys 

from vispy import scene
from vispy.color import get_colormap, ColorArray
from vispy.visuals import transforms

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QSpinBox, QShortcut
from PyQt5.QtWidgets import QTreeWidgetItem, QTreeWidget
from PyQt5.QtWidgets import QWidget, QFrame
from PyQt5.QtWidgets import QBoxLayout, QHBoxLayout, QGridLayout

from PyQt5.QtCore import Qt

import numpy as np
import numpy.random as rd

#from ._tensor_rank import TensorData

# an observable
class TensorData(object):
    def __init__(self):
        self.reset()
        self.observers = []

    def reset(self):
        self.tensor = None
        self.partitions = (3, 3, 3)
        self.ranks = []
        self.rank_viewmode = "hybrid"

    #TODO:
    # rank is a number in [0, 100]
    def rank(self, i, j, k):
        if self.tensor == 0:
            return 100 if i == j and j == k else 0
        elif self.tensor == 1:
            return 100 if i == 1 else 50
        else:
            return 50 if i == 1 or j == 1 else 0

    def sub_size(self, i, j, k):
        return (7, 3, 5)

    def setTensor(self, tensor):
        self.tensor = tensor

    def register(self, a_observer):
        self.observers.append(a_observer)

    # ValueError only if program error
    def unregister(self, a_observer):
        self.observers.remove(a_observer)

    def update(self):
        self._update_ranks()
        for ob in self.observers:
            ob.on_tensor_data_update(self)

    def _update_ranks(self):
        '''
        for given tensor, partitions, compute all involed ranks
        '''
        ...

_tensor_data = TensorData()

class HybridShowerUnit(scene.visuals.Markers):
    def __init__(self, rank, which_pos, how_many_seg, to_scale):
        super().__init__()
        poss = np.random.random(size=(10000, 3)) if rank > 50 else np.random.random(size=(10, 3))
        poss += which_pos
        poss -= how_many_seg//2
        poss *= to_scale

        self.set_data(poss, face_color=(1, 0, 0, 0.5) if rank > 50 else (1, 1, 1, 0.5), size=5)

class PointCloudShowerUnit(scene.visuals.Markers):
    def __init__(self, rank, which_pos, how_many_seg, to_scale):
        super().__init__()
        poss = np.random.random(size=(10000, 3)) if rank > 50 else np.random.random(size=(10, 3))
        poss += which_pos
        poss -= how_many_seg//2
        poss *= to_scale

        self.set_data(poss, size=5)

class PureColorShowerUnit(scene.visuals.Sphere):
    def __init__(self, rank, which_pos, how_many_seg, to_scale):
        super().__init__(radius=0.075
                , method='latitude'
                , color=(1, 0, 0, 0.5) if rank > 50 else (1, 1, 1, 0.5)
                )
        self.transform = transforms.MatrixTransform()
        self.transform.translate( to_scale * (which_pos - how_many_seg//2))


def get_shower_unit(rank_viewmode, *args):
    storage = {
            'hybrid': HybridShowerUnit,
            'point_cloud': PointCloudShowerUnit,
            'pure_color': PureColorShowerUnit,
            }
    try:
        return storage[rank_viewmode](*args)
    except KeyError:
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


    def _clear_all_shower_units(self):
        for unit in self.shower_units:
            unit.parent = None
        self.shower_units = []
    
    def on_tensor_data_update(self, tensor_data):
        self._clear_all_shower_units()
        to_scale = 0.4
        for l in range(tensor_data.partitions[0]):
            for m in range(tensor_data.partitions[1]):
                for n in range(tensor_data.partitions[2]):
                    shower_unit = get_shower_unit(
                              tensor_data.rank_viewmode
                            , tensor_data.rank(l, m, n)
                            , np.array((l, m, n))
                            , np.array(tensor_data.partitions)
                            , to_scale
                            )
                    self.shower_units.append(shower_unit)
                    self.view.add(shower_unit)

def g_keypress_manager(e):
    global _win
    if e.key() == Qt.Key_Escape:
        _win.close()


class EnhancedRadioButton(QRadioButton):
    def __init__(self, *args, **kv):
        super().__init__(*args, **kv)
        self.left = None 
        self.right = None
        self.setFocusPolicy(Qt.ClickFocus)
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left and self.left:
            self.setChecked(False)
            self.left.setChecked(True)
            self.left.setFocus(True)
        elif e.key() == Qt.Key_Right and self.right:
            self.setChecked(False)
            self.right.setChecked(True)
            self.right.setFocus(True)
        else:
            g_keypress_manager(e)


class TensorRankViewSelector(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('QFrame {border:1px solid;}')

        layout = QHBoxLayout(self)

        #point_cloud = QRadioButton('Point Cloud')
        point_cloud = EnhancedRadioButton('Point Cloud')
        point_cloud.toggled.connect(self.point_cloud_clicked_handler)
        layout.addWidget(point_cloud)

        #pure_color = QRadioButton('Pure Color')
        pure_color = EnhancedRadioButton('Pure Color')
        pure_color.toggled.connect(self.pure_color_clicked_handler)
        layout.addWidget(pure_color)

        #hybrid = QRadioButton('Hybrid')
        hybrid = EnhancedRadioButton('Hybrid')
        hybrid.toggled.connect(self.hybrid_clicked_handler)
        hybrid.setChecked(True)
        layout.addWidget(hybrid)

        all_choices = [point_cloud, pure_color, hybrid]
        for i in range(len(all_choices)):
            all_choices[i].left = all_choices[i-1] if i != 0 else all_choices[-1]
            all_choices[i].right = all_choices[i+1] if i + 1 != len(all_choices) else all_choices[0]
    
    def keyPressEvent(self, e):
        #if event.key() == Qt.Key_Left:# and self.left:
        print("OK")

    def hybrid_clicked_handler(self, enabled):
        if not enabled: return 
        _tensor_data.rank_viewmode = "hybrid"
        _tensor_data.update()

    def point_cloud_clicked_handler(self, enabled):
        if not enabled: return 
        _tensor_data.rank_viewmode = "point_cloud"
        _tensor_data.update()

    def pure_color_clicked_handler(self, enabled):
        if not enabled: return 
        _tensor_data.rank_viewmode = "pure_color"
        _tensor_data.update()


class TensorRankShower(QFrame):
    def __init__(self, tensor_data):
        super().__init__()
        tensor_data.register(self)
        self.setStyleSheet('QFrame {border:1px solid;}')

        self.layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        label_container = QWidget(self)
        self.layout.addWidget(label_container)
        label_container_layout = QHBoxLayout(label_container)
        label = QLabel('Tensor Rank', self)
        label.setAlignment(Qt.AlignHCenter)
        label_container_layout.addWidget(label)
        self.show_all_button = QPushButton('Expand/Collapse', self)
        label_container_layout.addWidget(self.show_all_button)
        self.tw = None

        def toggle_expand():
            if not self.tw: return
            self.tw.collapseAll() if self.is_expanded else self.tw.expandAll()
            self.is_expanded = not self.is_expanded

        self.show_all_button.clicked.connect(toggle_expand)
        

    def on_tensor_data_update(self, tensor_data):
        #for x in range(self.tw.topLevelItemCount()):
        #    pass
        if self.tw: self.tw.deleteLater()

        tw = QTreeWidget(self)
        self.layout.addWidget(tw)
        tw.setColumnCount(2)
        tw.setHeaderLabels(["where", "rank", "size"])

        for x in range(tensor_data.partitions[0]):
            x_child = QTreeWidgetItem(["x:{}".format(x)])
            #x_child.setText(0, "Hello")
            for y in range(tensor_data.partitions[1]):
                y_child = QTreeWidgetItem(["y:{}".format(y)])
                for z in range(tensor_data.partitions[2]):
                    sub_x, sub_y, sub_z = tensor_data.sub_size(x, y, z)
                    z_child = QTreeWidgetItem(["z:{}".format(z)
                        , "{}".format(tensor_data.rank(x, y, z))
                        , "{}x{}x{}".format(sub_x, sub_y, sub_z)
                        ])
                    y_child.addChild(z_child)
                x_child.addChild(y_child)
            tw.addTopLevelItem(x_child)

        self.tw = tw
        self.is_expanded = False


class XYZRangeSelector(QFrame):
    def __init__(self, tensor_data, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QFrame {border:1px solid;}')
        tensor_data.register(self)

        layout = QGridLayout(self)

        modes = ['X', 'Y', 'Z']
        self.how_many_partition_box = {}
        for m in range(1, 4):
            mode = modes[m - 1]

            label = QLabel(mode, self)
            label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(label, 1, m)

            self.how_many_partition_box[mode] = QSpinBox(self)
            self.how_many_partition_box[mode].setToolTip("how many partition?")
            layout.addWidget(self.how_many_partition_box[mode], 2, m)
        
            start_box = QSpinBox(self)
            start_box.setToolTip("start from")
            layout.addWidget(start_box, 3, m)

            end_box = QSpinBox(self)
            end_box.setToolTip("end at")
            layout.addWidget(end_box, 4, m)

        rerender_button = QPushButton('Rerender')
        rerender_button.clicked.connect(self.on_rerender_demand)
        layout.addWidget(rerender_button, 5, 3)
        self.layout = layout

    def on_tensor_data_update(self, tensor_data):
        layout = self.layout
        for i in range(1, 4):
            layout.itemAtPosition(2, i).widget().setValue(tensor_data.partitions[i-1])

            layout.itemAtPosition(3, i).widget().setValue(0)
            layout.itemAtPosition(4, i).widget().setValue(tensor_data.partitions[i-1] - 1)


    def on_rerender_demand(self):
        #TODO: pretend to collect data, and inform _tensor_data
        _tensor_data.partitions = [int(box.value()) for box in self.how_many_partition_box.values()]
        _tensor_data.setTensor(0)
        _tensor_data.update()
            
class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.resize(1200, 800)

        tensor_display_context = TensorDisplayContext(_tensor_data)

        def canvas_on_close_handler(_1, _2):
            self.close()
        tensor_display_context.canvas.set_on_close_handler(canvas_on_close_handler)
        box.addWidget(tensor_display_context.canvas.native)
        
        rightBoxWidget = QWidget()
        rightBox = QBoxLayout(QBoxLayout.TopToBottom, rightBoxWidget)
        rightBox.setAlignment(Qt.AlignTop)
        box.addWidget(rightBoxWidget)

        tensor_view_selector = TensorRankViewSelector()
        rightBox.addWidget(tensor_view_selector)

        xyzRangeSelector = XYZRangeSelector(_tensor_data)
        rightBox.addWidget(xyzRangeSelector)

        tw_container = TensorRankShower(_tensor_data)
        rightBox.addWidget(tw_container)

#        esc_shortcut = QShortcut(self)
#        esc_shortcut.activated.connect(self.close)
        
        self.show()

    def keyPressEvent(self, event):
        g_keypress_manager(event)
#        if event.key() == Qt.Key_Escape:
#            self.close()


_qt_app = QApplication([])
_win = Window()


def visualize(tensor):
    _tensor_data.reset()
    _tensor_data.setTensor(tensor)
    _tensor_data.update()

    _qt_app.exec_()

if __name__ == '__main__':
    visualize(1)
