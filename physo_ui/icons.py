"""Small vector icon set drawn with QPainter (no external assets required)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from . import theme


def _pen(color: str | QColor, width: float = 1.8) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def paint_icon(painter: QPainter, kind: str, rect: QRectF, color: str | QColor = theme.INK_2) -> None:
    """Draw a 24x24-ish line icon inside *rect*."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.topLeft())
    scale = min(rect.width(), rect.height()) / 24.0
    painter.scale(scale, scale)
    painter.setPen(_pen(color, 1.8))
    painter.setBrush(Qt.NoBrush)
    c = QColor(color)

    if kind == "data":
        painter.drawEllipse(QRectF(4, 4, 16, 5.5))
        path = QPainterPath()
        path.moveTo(4, 6.7)
        path.lineTo(4, 17.5)
        path.cubicTo(4, 20.2, 20, 20.2, 20, 17.5)
        path.lineTo(20, 6.7)
        painter.drawPath(path)
        painter.drawArc(QRectF(4, 8.2, 16, 5.5), 180 * 16, 180 * 16)
        painter.drawArc(QRectF(4, 13.0, 16, 5.5), 180 * 16, 180 * 16)
    elif kind == "model":
        for y, knob in ((6.5, 15.0), (12.0, 9.0), (17.5, 13.0)):
            painter.drawLine(QPointF(4, y), QPointF(20, y))
            painter.setBrush(c)
            painter.drawEllipse(QPointF(knob, y), 2.2, 2.2)
            painter.setBrush(Qt.NoBrush)
    elif kind == "train":
        painter.drawEllipse(QRectF(3.5, 3.5, 17, 17))
        path = QPainterPath()
        path.moveTo(10.0, 8.2)
        path.lineTo(16.2, 12.0)
        path.lineTo(10.0, 15.8)
        path.closeSubpath()
        painter.setBrush(c)
        painter.drawPath(path)
    elif kind == "result":
        painter.drawLine(QPointF(4, 20), QPointF(20, 20))
        for x, top in ((6.0, 13.0), (11.0, 8.0), (16.0, 4.5)):
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, top, 3.2, 20 - top - 1), 1.4, 1.4)
            painter.setBrush(c)
            painter.drawPath(path)
    elif kind == "predict":
        painter.drawLine(QPointF(5, 19), QPointF(18.5, 5.5))
        painter.drawLine(QPointF(11.5, 5.5), QPointF(18.5, 5.5))
        painter.drawLine(QPointF(18.5, 5.5), QPointF(18.5, 12.5))
        painter.setBrush(c)
        painter.drawEllipse(QPointF(5.5, 18.5), 1.6, 1.6)
    elif kind == "upload":
        painter.drawLine(QPointF(12, 15.5), QPointF(12, 4.5))
        painter.drawLine(QPointF(7.6, 8.6), QPointF(12, 4.2))
        painter.drawLine(QPointF(16.4, 8.6), QPointF(12, 4.2))
        path = QPainterPath()
        path.moveTo(4.5, 14.5)
        path.lineTo(4.5, 19.0)
        path.lineTo(19.5, 19.0)
        path.lineTo(19.5, 14.5)
        painter.drawPath(path)
    elif kind == "folder":
        path = QPainterPath()
        path.moveTo(3.5, 7.0)
        path.lineTo(3.5, 18.5)
        path.lineTo(20.5, 18.5)
        path.lineTo(20.5, 8.0)
        path.lineTo(11.0, 8.0)
        path.lineTo(9.5, 5.0)
        path.lineTo(3.5, 5.0)
        path.closeSubpath()
        painter.drawPath(path)
    elif kind == "save":
        path = QPainterPath()
        path.addRoundedRect(QRectF(4, 4, 16, 16), 3, 3)
        painter.drawPath(path)
        painter.drawRect(QRectF(8, 4, 8, 5.5))
        painter.drawRoundedRect(QRectF(7.5, 13, 9, 6), 1.2, 1.2)
    elif kind == "copy":
        painter.drawRoundedRect(QRectF(8.5, 3.5, 11, 12), 2, 2)
        painter.drawRoundedRect(QRectF(4.5, 8.5, 11, 12), 2, 2)
    elif kind == "sparkles":
        path = QPainterPath()
        path.moveTo(12, 3.2)
        path.lineTo(13.8, 9.2)
        path.lineTo(19.8, 11)
        path.lineTo(13.8, 12.8)
        path.lineTo(12, 18.8)
        path.lineTo(10.2, 12.8)
        path.lineTo(4.2, 11)
        path.lineTo(10.2, 9.2)
        path.closeSubpath()
        painter.setBrush(c)
        painter.drawPath(path)
        painter.drawEllipse(QPointF(18.5, 18.0), 1.5, 1.5)
    elif kind == "stop":
        painter.setBrush(c)
        painter.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2.2, 2.2)
    elif kind == "check":
        painter.drawLine(QPointF(5, 12.5), QPointF(10, 17.2))
        painter.drawLine(QPointF(10, 17.2), QPointF(19.5, 6.8))
    elif kind == "close":
        painter.drawLine(QPointF(6, 6), QPointF(18, 18))
        painter.drawLine(QPointF(18, 6), QPointF(6, 18))
    elif kind == "minimize":
        painter.drawLine(QPointF(5, 12), QPointF(19, 12))
    elif kind == "maximize":
        painter.drawRoundedRect(QRectF(5.5, 5.5, 13, 13), 1.5, 1.5)
    elif kind == "info":
        painter.drawEllipse(QRectF(3.5, 3.5, 17, 17))
        painter.drawLine(QPointF(12, 10.5), QPointF(12, 17))
        painter.setBrush(c)
        painter.drawEllipse(QPointF(12, 7.2), 1.1, 1.1)
    elif kind == "table":
        painter.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)
        painter.drawLine(QPointF(3.5, 9.5), QPointF(20.5, 9.5))
        painter.drawLine(QPointF(9.5, 9.5), QPointF(9.5, 19.5))
        painter.drawLine(QPointF(15.5, 9.5), QPointF(15.5, 19.5))
    else:
        painter.drawEllipse(QRectF(4, 4, 16, 16))
    painter.restore()


def icon(kind: str, color: str = theme.INK_2, size: int = 20) -> QIcon:
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    paint_icon(p, kind, QRectF(0, 0, size, size), color)
    p.end()
    return QIcon(pm)
