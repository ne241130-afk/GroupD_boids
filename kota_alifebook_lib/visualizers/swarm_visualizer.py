import numpy as np
import vispy
from vispy.scene import SceneCanvas
from vispy.scene import visuals


class SwarmVisualizer(object):
    """個体数・矢印色・マーカー色を指定できる群れ用ビジュアライザ。

    ``agent_count`` は描画するアリ数の検証に使う。個体数を固定しない
    場合は ``None`` のままにする。色は RGB または RGBA のタプル／配列を
    与えられ、マーカーごとに異なる色・サイズを指定することもできる。
    """
    ARROW_SIZE = 20

    def __init__(self, width=600, height=600, agent_count=None,
                 arrow_color=(0.85, 0.85, 0.85, 1.0), arrow_size=12,
                 show_arrows=True,
                 title="Kota ALife SwarmVisualizer"):
        self.agent_count = agent_count
        self.arrow_color = arrow_color
        self.arrow_size = arrow_size
        self.show_arrows = show_arrows
        self._canvas = SceneCanvas(size=(width, height), position=(0, 0),
                                   keys='interactive', title=title)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = 'turntable'
        self._axis = visuals.XYZAxis(parent=self._view.scene)
        self._arrows = None
        self._markers = None
        self._canvas.show()

    def update(self, position, direction, arrow_color=None):
        """矢印を更新する。``arrow_color`` はフレーム単位で上書き可能。"""
        assert position.ndim == 2 and position.shape[1] in (2, 3)
        assert direction.ndim == 2 and direction.shape[1] in (2, 3)
        assert position.shape[0] == direction.shape[0]
        if self.agent_count is not None and position.shape[0] != self.agent_count:
            raise ValueError("position の個体数が agent_count と一致しません")
        if self.show_arrows:
            if self._arrows is None:
                self._arrows = visuals.Arrow(arrow_size=self.arrow_size,
                                             arrow_type='triangle_30',
                                             parent=self._view.scene)
            # arrow_coordinate[0::2] is position of arrow and
            # arrow_coordinate[1::2] is direction of tail (length is ignored)
            arrow_coordinate = np.repeat(position, 2, axis=0)
            arrow_coordinate[::2] -= direction
            self._arrows.set_data(
                arrows=arrow_coordinate.reshape((-1, 6)),
                color=self.arrow_color if arrow_color is None else arrow_color,
            )
        self._canvas.update()
        vispy.app.process_events()

    def set_markers(self, position, face_color=(1, 0, 0, 1), size=20):
        """マーカーを描画／更新する。

        ``face_color`` と ``size`` には単一値のほか、各マーカーに対応する
        配列を渡せるため、巣・餌・アリなどを任意の色と大きさで区別できる。
        """
        assert position.ndim == 2 and position.shape[-1] in (2, 3)
        if self._markers is None:
            self._markers = visuals.Markers(parent=self._view.scene)
        self._markers.set_data(position, face_color=face_color, size=size)
        self._canvas.update()
        vispy.app.process_events()

    def __bool__(self):
        return not self._canvas._closed


if __name__ == '__main__':
    N = 1000
    import numpy as np

    v = SwarmVisualizer(agent_count=N, arrow_color=(0.2, 0.8, 1.0, 1.0))
    pos = np.random.normal(size=(N, 3), scale=0.2)
    vel = np.random.normal(size=(N, 3), scale=0.2) * 0.001
    v.set_markers(np.array([[0, 0, 0]]), face_color=(1, 0, 0, 1), size=20)
    while v:
        vel -= pos * 0.00001
        pos +=  vel
        v.update(pos, vel)
