import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QCheckBox, QFileDialog, QColorDialog, QMessageBox, QDialog, QHBoxLayout, QSlider, QVBoxLayout, QTextEdit, QStyle, QSizePolicy, QLineEdit, QToolButton, QComboBox
from PyQt6.QtGui import QPixmap, QPainter, QColor, QRegion, QIcon, QImage, QPixmap, QGuiApplication, QFont, QDesktopServices, QPen
from PyQt6.QtCore import Qt, QSettings, QSize, pyqtSignal, QUrl, QLoggingCategory, QUrl
from PyQt6.QtMultimedia import QSoundEffect
import cv2
import os
import numpy as np
from datetime import datetime  
import json
import random
import math
import networkx as nx
import tempfile, shutil, zipfile
import csv

QLoggingCategory.setFilterRules("*.debug=false\n*.info=false\n*.warning=false\n*.critical=false")

cor_base_salva = None
cor_sobreposta_salva = None

def cv2_to_qpixmap(cv_img):
    """Converte uma imagem OpenCV (BGR) para QPixmap."""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = 3 * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def inverter_cor(rgb_tuple):
    """Retorna o inverso de uma cor RGB (ex: (0, 255, 0) vira (255, 0, 255))"""
    return tuple(255 - c for c in rgb_tuple)

def crossing_number(img, x, y):
    """
    Calcula o Crossing Number para o pixel (x,y) em uma imagem binária.
    """
    p = []
    p.append(1 if img[y-1, x] > 0 else 0)     
    p.append(1 if img[y-1, x+1] > 0 else 0)    
    p.append(1 if img[y, x+1] > 0 else 0)      
    p.append(1 if img[y+1, x+1] > 0 else 0)    
    p.append(1 if img[y+1, x] > 0 else 0)     
    p.append(1 if img[y+1, x-1] > 0 else 0)    
    p.append(1 if img[y, x-1] > 0 else 0)      
    p.append(1 if img[y-1, x-1] > 0 else 0)  
    p.append(p[0])  

    cn = 0
    for i in range(8):
        if p[i] != p[i+1]:
            cn += 1
    return cn / 2

def compute_minutia_direction(skeleton, x, y, max_length=20):
    """
    Retorna a direção aproximada da crista a partir de (x,y) até um ponto adiante.
    """
    h, w = skeleton.shape
    path = []
    visited = set()
    stack = [(x, y)]

    while stack and len(path) < max_length:
        cx, cy = stack.pop()
        if (cx, cy) in visited or cx <= 0 or cy <= 0 or cx >= w-1 or cy >= h-1:
            continue
        visited.add((cx, cy))

        if skeleton[cy, cx] != 255:
            continue

        path.append((cx, cy))

        # 8 vizinhos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    nx, ny = cx + dx, cy + dy
                    if skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                        stack.append((nx, ny))

    if len(path) >= 2:
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        norm = math.hypot(dx, dy)
        return (dx / norm, dy / norm) if norm != 0 else (0, -1)
    return (0, -1)

def process_minutiae_trabalho(image, cor=(0, 255, 0), raio=8, pontos_predefinidos=None, quantidade=25, distancia_threshold=60):
    """
    image: imagem BGRA (com transparência)
    cor: tupla RGB ou BGR (ex: (0, 255, 0) = verde)
    raio: raio do círculo desenhado em cada minúcia
    pontos_predefinidos: lista opcional de minúcias já detectadas (x, y, score)
    """
    directions = []
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        binary = np.zeros_like(alpha)
        binary[alpha > 0] = 255
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    if pontos_predefinidos is not None:
        filtered = pontos_predefinidos
    else:
        skeleton = cv2.ximgproc.thinning(binary)

        y_coords, x_coords = np.where(skeleton == 255)
        minutiae = []
        margin = 20
        distance_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        min_distance = 4  

        for x, y in zip(x_coords, y_coords):
            if y - 1 < 0 or y + 1 >= skeleton.shape[0] or x - 1 < 0 or x + 1 >= skeleton.shape[1]:
                continue

            if distance_transform[y, x] < min_distance:
                continue  

            cn = crossing_number(skeleton, x, y)
            if cn in [1, 3]:
                if margin < x < skeleton.shape[1] - margin and margin < y < skeleton.shape[0] - margin:
                    score = 0.7 if cn == 1 else 0.9
                    minutiae.append((x, y, score))

        filtered = []
        distance_threshold = distancia_threshold
        sorted_minutiae = sorted(minutiae, key=lambda m: m[2], reverse=True)
        for m in sorted_minutiae:
            x, y, score = m
            if not filtered:
                filtered.append(m)
            else:
                dist = np.hypot(np.array([x - fx for fx, _, _ in filtered]),
                                np.array([y - fy for _, fy, _ in filtered]))
                if np.all(dist > distance_threshold):
                    filtered.append(m)
            if len(filtered) == quantidade:
                break
            
            directions.append(compute_minutia_direction(skeleton, x, y))

    resultado = image.copy()
    for x, y, _ in filtered:
        if resultado.shape[2] == 4:
            b, g, r = map(int, cor) 
            cor_com_alpha = (b, g, r, 255)
            cv2.circle(resultado, (x, y), raio, cor_com_alpha, -1)
        else:
            cv2.circle(resultado, (x, y), raio, cor, -1)

    return resultado, filtered, directions

def _extrair_mascara_cristas(image):
    if image is None:
        return None

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        alpha_ratio = np.count_nonzero(alpha) / float(alpha.size)
        if 0.01 < alpha_ratio < 0.98:
            return ((alpha > 0) * 255).astype(np.uint8)

    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = gray[threshold == 255]
    black_pixels = gray[threshold == 0]
    if white_pixels.size == 0 or black_pixels.size == 0:
        return ((gray > 0) * 255).astype(np.uint8)

    fundo_claro = np.mean(white_pixels) > np.mean(black_pixels)
    if fundo_claro:
        threshold = 255 - threshold
    return threshold.astype(np.uint8)

def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _orientation_blocks(binary, block_size=16):
    h, w = binary.shape[:2]
    grid_h = h // block_size
    grid_w = w // block_size
    if grid_h < 3 or grid_w < 3:
        return None, None

    work = binary[:grid_h * block_size, :grid_w * block_size]
    gray = cv2.GaussianBlur(work.astype(np.float32) / 255.0, (5, 5), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    theta = np.zeros((grid_h, grid_w), dtype=np.float32)
    coherence = np.zeros((grid_h, grid_w), dtype=np.float32)
    density = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        y0 = gy * block_size
        y1 = y0 + block_size
        for gx in range(grid_w):
            x0 = gx * block_size
            x1 = x0 + block_size
            sl = np.s_[y0:y1, x0:x1]

            gxx = float(np.sum(grad_x[sl] * grad_x[sl]))
            gyy = float(np.sum(grad_y[sl] * grad_y[sl]))
            gxy = float(np.sum(grad_x[sl] * grad_y[sl]))

            theta[gy, gx] = np.mod(0.5 * np.arctan2(-2 * gxy, gxx - gyy) + np.pi / 2, np.pi)
            coherence[gy, gx] = np.sqrt((gxx - gyy) ** 2 + (2 * gxy) ** 2) / (gxx + gyy + 1e-6)
            density[gy, gx] = np.mean(work[sl] > 0)

    ksize = 5 if min(theta.shape) >= 5 else 3
    cos2 = cv2.GaussianBlur(np.cos(2 * theta), (ksize, ksize), 0)
    sin2 = cv2.GaussianBlur(np.sin(2 * theta), (ksize, ksize), 0)
    theta = np.mod(0.5 * np.arctan2(sin2, cos2), np.pi)

    valid = (density > 0.05) & (density < 0.95) & (coherence > 0.05)
    return theta, valid

def _poincare_candidates(theta, valid, block_size):
    if theta is None or valid is None:
        return []

    grid_h, grid_w = theta.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    candidates = []

    for y in range(1, grid_h - 1):
        for x in range(1, grid_w - 1):
            if not valid[y, x]:
                continue

            values = []
            ok = True
            for dy, dx in neighbors:
                if not valid[y + dy, x + dx]:
                    ok = False
                    break
                values.append(theta[y + dy, x + dx] * 2)

            if not ok:
                continue

            total = 0.0
            for idx in range(len(values)):
                total += _wrap_pi(values[(idx + 1) % len(values)] - values[idx])

            poincare_index = total / (4 * np.pi)
            if abs(poincare_index) >= 0.35:
                candidates.append(((x + 0.5) * block_size, (y + 0.5) * block_size, poincare_index))

    return candidates

def _cluster_singular_candidates(candidates, block_size):
    clusters = []
    merge_distance = max(2.5 * block_size, 20)

    for x, y, index in sorted(candidates, key=lambda p: abs(p[2]), reverse=True):
        weight = abs(index) + 1e-6
        assigned = False
        for cluster in clusters:
            mesmo_sinal = cluster["index_sum"] == 0 or index * cluster["index_sum"] > 0
            if mesmo_sinal and math.hypot(x - cluster["x"], y - cluster["y"]) <= merge_distance:
                total_weight = cluster["weight"] + weight
                cluster["x"] = (cluster["x"] * cluster["weight"] + x * weight) / total_weight
                cluster["y"] = (cluster["y"] * cluster["weight"] + y * weight) / total_weight
                cluster["weight"] = total_weight
                cluster["index_sum"] += index
                cluster["hits"] += 1
                assigned = True
                break

        if not assigned:
            clusters.append({
                "x": float(x),
                "y": float(y),
                "weight": float(weight),
                "index_sum": float(index),
                "hits": 1,
            })

    return sorted(clusters, key=lambda c: abs(c["index_sum"]), reverse=True)

def _select_singularities_by_geometry(clusters):
    top = clusters[:8]
    if len(top) < 3:
        return [], []

    best = None
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            for k in range(len(top)):
                if k in (i, j):
                    continue

                c1, c2, d = top[i], top[j], top[k]
                core_dist = math.hypot(c1["x"] - c2["x"], c1["y"] - c2["y"])
                delta_dist_1 = math.hypot(c1["x"] - d["x"], c1["y"] - d["y"])
                delta_dist_2 = math.hypot(c2["x"] - d["x"], c2["y"] - d["y"])

                if core_dist <= 1:
                    continue

                score = (
                    abs(c1["index_sum"]) + abs(c2["index_sum"]) + abs(d["index_sum"])
                    + 0.05 * min(delta_dist_1, delta_dist_2)
                    - 0.04 * core_dist
                )
                if min(delta_dist_1, delta_dist_2) < core_dist * 1.2:
                    score -= 100

                if best is None or score > best[0]:
                    best = (score, [c1, c2], [d])

    if best is None:
        return [], []
    return best[1], best[2]

def _clusters_to_points(clusters, scale, width, height, limit):
    points = []
    for cluster in clusters[:limit]:
        x = int(round(cluster["x"] / scale))
        y = int(round(cluster["y"] / scale))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        if not any(math.hypot(x - px, y - py) < 3 for px, py in points):
            points.append((x, y))
    return points

def _ordenar_pontos_por_referencia(points, reference=None):
    if not points:
        return []
    if reference is None:
        return sorted(points, key=lambda p: (p[0], p[1]))
    return sorted(points, key=lambda p: (math.hypot(p[0] - reference[0], p[1] - reference[1]), p[0], p[1]))

def detect_singularities_trabalho(image, max_size=512, block_size=16):
    binary = _extrair_mascara_cristas(image)
    if binary is None or np.count_nonzero(binary) < 200:
        return [], []

    height, width = binary.shape[:2]
    scale = min(1.0, float(max_size) / float(max(height, width)))
    small_width = max(1, int(width * scale))
    small_height = max(1, int(height * scale))

    if small_width < block_size * 3 or small_height < block_size * 3:
        return [], []

    if scale < 1.0:
        small = cv2.resize(binary, (small_width, small_height), interpolation=cv2.INTER_AREA)
    else:
        small = binary.copy()
    small = ((small > 32) * 255).astype(np.uint8)

    theta, valid = _orientation_blocks(small, block_size=block_size)
    candidates = _poincare_candidates(theta, valid, block_size)
    clusters = _cluster_singular_candidates(candidates, block_size)

    core_clusters = [c for c in clusters if c["index_sum"] < 0][:2]
    delta_clusters = [c for c in clusters if c["index_sum"] > 0][:1]

    if len(core_clusters) < 2 or len(delta_clusters) < 1:
        geo_cores, geo_delta = _select_singularities_by_geometry(clusters)
        if len(core_clusters) < 2 and geo_cores:
            core_clusters = geo_cores[:2]
        if len(delta_clusters) < 1 and geo_delta:
            delta_clusters = geo_delta[:1]

    delta_points = _clusters_to_points(delta_clusters, scale, width, height, 1)
    core_points = _clusters_to_points(core_clusters, scale, width, height, 2)
    core_points = _ordenar_pontos_por_referencia(core_points, delta_points[0] if delta_points else None)

    return core_points, delta_points

def reaplicar_cor(self):
    """Reaplica as cores previamente salvas nas imagens carregadas, garantindo a consistência das edições."""
    cor_base_salva = self.settings.value("cor_base", None)
    cor_sobreposta_salva = self.settings.value("cor_sobreposta", None)

    if cor_base_salva and self.base_scaled:
        cor_base_atual = QColor(cor_base_salva)
        if not self.base_colored or cor_base_atual != self.cor_base_salva:
            self.cor_base_salva = cor_base_atual
            painter = QPainter(self.base_scaled)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(self.base_scaled.rect(), cor_base_atual)
            painter.end()
            self.base_colored = self.base_scaled.copy()

    if cor_sobreposta_salva and self.sobreposta_scaled:
        cor_dialog = CorDialog(self)
        self.sobreposta_colored = cor_dialog.aplicar_cor_na_imagem(
            self.sobreposta_scaled, QColor(cor_sobreposta_salva)
        )
        self.sobreposta_scaled = self.sobreposta_colored.copy()

    self.update()

def resource_path(relative_path):
    """Retorna o caminho correto de um recurso, garantindo compatibilidade ao rodar o software como executável."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def create_colored_icon(icon, color, size=64):
    """Cria e retorna um ícone com a cor personalizada aplicada, mantendo transparência e suavização."""
    pixmap = QPixmap(size, size)  
    pixmap.fill(Qt.GlobalColor.transparent)  

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)  
    icon.paint(painter, pixmap.rect())  
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color) 
    painter.end()
    return QIcon(pixmap)

def posicionar_dialogo_no_parent(dialog, parent, centralizar=False):
    if parent is None:
        return

    parent_window = parent.window() if parent.window() else parent
    screen = parent_window.screen()
    if screen is None:
        screen = QGuiApplication.screenAt(parent_window.frameGeometry().center())
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return

    available = screen.availableGeometry()
    if centralizar:
        dialog.adjustSize()
    dialog_width = max(1, dialog.width())
    dialog_height = max(1, dialog.height())

    if centralizar:
        center = parent_window.frameGeometry().center()
        x = center.x() - dialog_width // 2
        y = center.y() - dialog_height // 2
    else:
        geom = dialog.geometry()
        x = available.left() + max(0, geom.x())
        y = available.top() + max(0, geom.y())

    x = max(available.left(), min(x, available.right() - dialog_width + 1))
    y = max(available.top(), min(y, available.bottom() - dialog_height + 1))
    dialog.move(x, y)

def escolher_arquivo(parent, titulo, filtro, diretorio=""):
    dialog = QFileDialog(parent, titulo, diretorio, filtro)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    posicionar_dialogo_no_parent(dialog, parent, centralizar=True)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        arquivos = dialog.selectedFiles()
        return arquivos[0] if arquivos else ""
    return ""

def escolher_arquivo_para_salvar(parent, titulo, filtro, diretorio=""):
    dialog = QFileDialog(parent, titulo, diretorio, filtro)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    posicionar_dialogo_no_parent(dialog, parent, centralizar=True)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        arquivos = dialog.selectedFiles()
        return arquivos[0] if arquivos else ""
    return ""

def escolher_cor(parent, cor_inicial):
    dialog = QColorDialog(cor_inicial, parent)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    posicionar_dialogo_no_parent(dialog, parent, centralizar=True)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.selectedColor()
    return QColor()

class Layout(QWidget):
    """ 
    Classe principal da interface gráfica do aplicativo. Gerencia a exibição das imagens base e sobreposta, além de fornecer ferramentas para alinhamento, rotação, escala e personalização de cores. Também inclui funcionalidades para salvar e carregar estados """
    imagem_carregada = pyqtSignal()  
    imagem_base_carregada = pyqtSignal()
    imagem_base_ok = pyqtSignal(bool)  
    imagem_sobreposta_carregada = pyqtSignal(bool)  

    def __init__(self, *args, **kwargs):
        """Inicializa a interface, configura os elementos gráficos e define os parâmetros iniciais."""
        super().__init__(*args, **kwargs)
        
        self.setWindowTitle("BENAH")

        self.settings = QSettings("UTFPR", "InfantID_App")

        screen_geometry = QGuiApplication.primaryScreen().geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()

        self.scale_x = self.screen_width / 1920
        self.scale_y = self.screen_height / 1080

        saved_color = self.settings.value("cor_fundo", QColor(71, 142, 213))  
        self.rect_color = QColor(saved_color)  

        self.base_colored = None
        self.sobreposta_colored = None

        self.transparencia = 0.5

        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)
        self.sobreposta_zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.base_position_x = int(10 * self.scale_x)
        self.base_position_y = int(115 * self.scale_y)
        self.sobreposta_position_x = int(750 * self.scale_x)
        self.sobreposta_position_y = int(115 * self.scale_y)

        self.base_pixmap = None  
        self.sobreposta_pixmap = None
        self.base_scaled = None  
        self.sobreposta_scaled = None

        self.cor_base_salva = None
        self.cor_sobreposta_salva = None

        self.cor_sobreposta_alterada = False  

        self.minucias_detectadas = False

        self.pontos_correspondentes_sobreposta = set()

        self.minucias_sobreposta = []
        self.cores_minucias_base = []
        self.cores_minucias_sobreposta = []

        self.modo_marcacao = None  
        self.pontos_delta = []
        self.pontos_core = []
        self.pontos_delta_sobreposta_detectados = []
        self.pontos_core_sobreposta_detectados = []
        self.pontos_minucia_manual = []
        self.ponto_editando = None

        self.grafo_base = []
        self.grafo_sobreposta = []
  
        self.espessura_grafo = 5
        self.transparencia_grafo = 1.0

        self.espessura_matching = 3
        self.grafo_matching = []

        self.mostrar_direcao_minucias = False
        self.tamanho_direcao_altura = 70
        self.tamanho_direcao_espessura = 5

        self.cor_minucias = (0, 255, 0)  
        self.cor_core = QColor(0, 255, 255)   
        self.cor_delta = QColor(255, 175, 75)   
        self.raio_minucias = 20         
        self.cor_marcacao = QColor(255, 255, 0)  
        self.cor_grafo = QColor(255, 255, 255, 180)  
        self.cor_direcao_minucias = QColor(255, 255, 255) 
        self.cor_matching = QColor(0, 255, 255) 

        cor_direcao_salva = self.settings.value("cor_direcao_minucias", None)
        if cor_direcao_salva:
            self.cor_direcao_minucias = QColor(cor_direcao_salva)

        cor_matching_salva = self.settings.value("cor_matching", None)
        if cor_matching_salva:
            self.cor_matching = QColor(cor_matching_salva)

        cor_grafo_salva = self.settings.value("cor_grafo", None)
        if cor_grafo_salva:
            self.cor_grafo = QColor(cor_grafo_salva)

        cor_minucias_salva = self.settings.value("cor_minucias", None)
            
        if cor_minucias_salva:
            self.cor_minucias = (cor_minucias_salva[2], cor_minucias_salva[1], cor_minucias_salva[0])

        cor_core_salva = self.settings.value("cor_core", None)
        if cor_core_salva:
            self.cor_core = QColor(cor_core_salva)

        cor_delta_salva = self.settings.value("cor_delta", None)
        if cor_delta_salva:
            self.cor_delta = QColor(cor_delta_salva)

        cor_marcacao_salva = self.settings.value("cor_marcacao", None)
        if cor_marcacao_salva:
            self.cor_marcacao = QColor(cor_marcacao_salva)

        self.mostrar_numeros_marcacao = self.settings.value("mostrar_numeros_marcacao", True, type=bool)

        cor_marcacao_salva = self.settings.value("cor_marcacao", None)
        if cor_marcacao_salva:
            self.cor_marcacao = QColor(cor_marcacao_salva)

        cor_base = self.settings.value("cor_base", None)
        cor_sobreposta = self.settings.value("cor_sobreposta", None)

        if cor_base:
            self.cor_base_salva = QColor(cor_base)
        if cor_sobreposta:
            self.cor_sobreposta_salva = QColor(cor_sobreposta)

        self.easter_egg_active = False

        self.minucia_selecionada = None

        self.janelas_abertas = []  
        self.max_janelas = 5  
        
        self.easter_egg_color = QColor("#fefefe")  

        self.easter_egg_image = QPixmap(resource_path("Complementos/Escudo-São-Paulo.png"))
        if self.easter_egg_image.isNull():
            print("Erro: Imagem do easter egg não foi carregada.")

        self.sound_effect = QSoundEffect()
        self.sound_effect.setSource(QUrl.fromLocalFile(resource_path("Complementos/hino_do_sao_paulo.wav")))
        self.sound_effect.setLoopCount(9999)

        self.check_base_imagem = QCheckBox(self)
        self.check_base_imagem.setGeometry(int(0 * self.scale_x), int(0 * self.scale_y), int(0 * self.scale_x), int(0 * self.scale_y))
        self.check_base_imagem.setEnabled(False)
        
        self.check_imagem_sobreposta = QCheckBox(self)
        self.check_imagem_sobreposta.setGeometry(int(0 * self.scale_x), int(0 * self.scale_y), int(0 * self.scale_x), int(0 * self.scale_y))
        self.check_imagem_sobreposta.setEnabled(False)

        self.check_minucias = QCheckBox(self)
        self.check_minucias.setGeometry(int(0 * self.scale_x), int(0 * self.scale_y), int(0 * self.scale_x), int(0 * self.scale_y))
        self.check_minucias.setEnabled(False)

        self.sidebar_widgets = []
        self.sidebar_x = 1705
        self.sidebar_width = 200

        self.btnmin = QPushButton(self)
        min_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton)
        self.btnmin.setIcon(create_colored_icon(min_icon, QColor("white"), 32))
        self.btnmin.setGeometry(int(1705 * self.scale_x), int(10 * self.scale_y), int(60 * self.scale_x), int(30 * self.scale_y))
        self.btnmin.clicked.connect(self.minimize_app)

        self.btnres = QPushButton(self)
        restore_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton)
        self.btnres.setIcon(create_colored_icon(restore_icon, QColor("white"), 32))
        self.btnres.setGeometry(int(1775 * self.scale_x), int(10 * self.scale_y), int(60 * self.scale_x), int(30 * self.scale_y))
        self.btnres.clicked.connect(self.resetar_zoom)

        self.btnClose = QPushButton(self)
        close_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        self.btnClose.setIcon(create_colored_icon(close_icon, QColor("white"), 32))
        self.btnClose.setGeometry(int(1845 * self.scale_x), int(10 * self.scale_y), int(60 * self.scale_x), int(30 * self.scale_y))
        self.btnClose.clicked.connect(self.close_app)

        self._criar_secao_lateral("Principal", 65)
        self.btn = QPushButton("Carregar Imagens", self)
        self._configurar_botao_lateral(
            self.btn,
            94,
            altura=48,
            destaque=True
        )
        self.btn.clicked.connect(self.carregar)

        self._criar_secao_lateral("Ferramentas", 170)
        self.btn_minu = QPushButton("Marcar", self)
        self._configurar_botao_lateral(self.btn_minu, 199, altura=44)
        self.btn_minu.clicked.connect(self.abrir_janela_detectar_minucias)

        self.btn_graf = QPushButton("Grafo", self)
        self._configurar_botao_lateral(self.btn_graf, 251, altura=44)
        self.btn_graf.clicked.connect(self.abrir_janela_grafo)

        self.matc = QPushButton("Matching", self)
        self._configurar_botao_lateral(self.matc, 303, altura=44)
        self.matc.clicked.connect(self.abrir_janela_matching)

        self._criar_secao_lateral("Ajustes", 383)
        self.btn2 = QPushButton("Cores", self)
        self._configurar_botao_lateral(self.btn2, 412, altura=44)
        self.btn2.clicked.connect(self.mudar_cor_do_retangulo)

        self.btn5 = QPushButton("Transparência", self)
        self._configurar_botao_lateral(self.btn5, 464, altura=44)
        self.btn5.clicked.connect(self.open_transparency_dialog)

        self.atr = QPushButton("Atributos", self)
        self._configurar_botao_lateral(self.atr, 516, altura=44)
        self.atr.clicked.connect(self.abrir_janela_atributos)
        
        self._criar_secao_lateral("Saída", 596)
        self.btn3 = QPushButton("Salvar", self)
        self._configurar_botao_lateral(
            self.btn3,
            625,
            altura=48,
            destaque=True
        )
        self.btn3.clicked.connect(self.salvar_imagem_e_dados)

        self.btn_aplicar_log = QPushButton("Log", self)
        self._configurar_botao_lateral(
            self.btn_aplicar_log,
            685,
            largura=95,
            altura=42
        )
        self.btn_aplicar_log.clicked.connect(self.abrir_janela_log)

        self.btn4 = QPushButton("Manual", self)
        self._configurar_botao_lateral(
            self.btn4,
            685,
            x=1810,
            largura=95,
            altura=42
        )
        self.btn4.clicked.connect(self.open_manual)

        pixmap = QPixmap(resource_path(os.path.join("Fotos", "UTFPR_biometria.png")))
        largura = int(165 * self.scale_x)
        altura = int(165 * self.scale_y)
        pixmap_resized = pixmap.scaled(largura, altura, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

        white_pixmap = QPixmap(pixmap_resized.size())
        white_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(white_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap_resized)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(white_pixmap.rect(), Qt.GlobalColor.white)
        painter.end()

        self.btn_site_utfpr = QPushButton(self)
        self.btn_site_utfpr.setIcon(QIcon(white_pixmap))
        self.btn_site_utfpr.setIconSize(QSize(largura, altura))
        self.btn_site_utfpr.setGeometry(int(1722 * self.scale_x), int(770 * self.scale_y), largura, altura)
        self.btn_site_utfpr.setStyleSheet("background-color: transparent; border: none;")
        self.btn_site_utfpr.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://sites.google.com/view/utfprbiometria")))

        pixmap = QPixmap(resource_path(os.path.join("Fotos", "CNPQ.png")))
        largura = int(190 * self.scale_x)
        altura = int(58 * self.scale_y)
        pixmap_resized = pixmap.scaled(largura, altura, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

        white_pixmap = QPixmap(pixmap_resized.size())
        white_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(white_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap_resized)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(white_pixmap.rect(), Qt.GlobalColor.white)
        painter.end()

        self.btn_imagem = QPushButton(self)
        self.btn_imagem.setIcon(QIcon(white_pixmap))
        self.btn_imagem.setIconSize(QSize(largura, altura))
        self.btn_imagem.setGeometry(int(1710 * self.scale_x), int(975 * self.scale_y), largura, altura)
        self.btn_imagem.setStyleSheet("background-color: transparent; border: none;")
        self.btn_imagem.clicked.connect(self.agradecimentos) 
            
        self.history = [] 
        self.current_index = -1  
        self.log_usuario = []
        self.log_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_inicio = datetime.now().isoformat(timespec="seconds")
        self.setMouseTracking(True)
        
        self.carregar_cor_fundo()
        self.showFullScreen()
        self.registrar_log("sessao_iniciada")

        cor_minucias_salva = self.settings.value("cor_minucias", None)
        if cor_minucias_salva:
            self.cor_minucias = (cor_minucias_salva[2], cor_minucias_salva[1], cor_minucias_salva[0])

    def _criar_secao_lateral(self, texto, y):
        label = QLabel(texto.upper(), self)
        label.setGeometry(
            int(self.sidebar_x * self.scale_x),
            int(y * self.scale_y),
            int(self.sidebar_width * self.scale_x),
            int(20 * self.scale_y)
        )
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet(
            f"background: transparent; color: #EAF3FF; font-size: {max(12, int(13 * self.scale_x))}px; font-weight: 800;"
        )
        linha = QLabel(self)
        linha.setGeometry(
            int(self.sidebar_x * self.scale_x),
            int((y + 24) * self.scale_y),
            int(self.sidebar_width * self.scale_x),
            max(1, int(1 * self.scale_y))
        )
        linha.setStyleSheet("background-color: #60A5FA;")
        self.sidebar_widgets.extend([label, linha])
        return label

    def _configurar_botao_lateral(self, botao, y, x=None, largura=None, altura=42, destaque=False):
        x = self.sidebar_x if x is None else x
        largura = self.sidebar_width if largura is None else largura
        botao.setGeometry(
            int(x * self.scale_x),
            int(y * self.scale_y),
            int(largura * self.scale_x),
            int(altura * self.scale_y)
        )
        botao.setCursor(Qt.CursorShape.PointingHandCursor)
        if destaque:
            botao.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1D4ED8;
                    color: #FFFFFF;
                    border: 1px solid #60A5FA;
                    border-radius: 5px;
                    padding: 5px;
                    font-size: {int(14 * self.scale_x)}px;
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    background-color: #2563EB;
                }}
                QPushButton:pressed {{
                    background-color: #1E40AF;
                    border: 1px solid #93C5FD;
                }}
            """)

    def atualizar_status_lateral(self):
        return

    def mudar_cor_minucias(self):
        """Permite ao usuário escolher uma cor personalizada para as minúcias."""
        cor = escolher_cor(self, QColor(*[int(c) for c in reversed(self.cor_minucias)]))
        if cor.isValid():
            self.cor_minucias = (cor.blue(), cor.green(), cor.red())
            
            self.settings.setValue("cor_minucias", (cor.red(), cor.green(), cor.blue()))
            self.settings.sync()
            
            if hasattr(self, 'minucias_pixmap') and self.minucias_pixmap:
                self.detectar_minutiae() 

    def detectar_minutiae(self):
        if not hasattr(self, 'base_pixmap') or self.base_pixmap is None:
            QMessageBox.warning(self, "Aviso", "Carregue uma imagem base primeiro!")
            return
        
        if not hasattr(self, 'sobreposta_pixmap') or self.sobreposta_pixmap is None:
            QMessageBox.warning(self, "Erro", "Carregue a imagem sobreposta antes de detectar pontos.")
            return

        self.base_original = self.base_pixmap.copy()

        qimage = self.base_original.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        imagem_cv_original = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))

        escala = 0.35
        imagem_redimensionada = cv2.resize(imagem_cv_original, (0, 0), fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

        quantidade = getattr(self, "quantidade_minucias", 20)
        distancia = getattr(self, "distancia_minima", 80)

        _, pontos_reduzidos, direcoes_reduzidas = process_minutiae_trabalho(
            imagem_redimensionada,
            cor=self.cor_minucias,
            raio=self.raio_minucias,
            quantidade=quantidade,
            distancia_threshold=distancia
        )

        pontos = [(int(x / escala), int(y / escala), score) for (x, y, score) in pontos_reduzidos]

        self.direcoes_minucias_base = {}
        for idx in range(len(pontos_reduzidos)):
            if idx < len(direcoes_reduzidas):
                self.direcoes_minucias_base[idx + 1] = direcoes_reduzidas[idx]
            else:
                self.direcoes_minucias_base[idx + 1] = (0, -1)

        imagem_minucias = np.zeros((height, width, 4), dtype=np.uint8)
        for x, y, _ in pontos:
            r, g, b = map(int, self.cor_minucias)
            cv2.circle(imagem_minucias, (x, y), self.raio_minucias, (b, g, r, 255), -1)

        qimage_minucias = QImage(imagem_minucias.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
        self.minucias_pixmap = QPixmap.fromImage(qimage_minucias)

        self.minucias_scaled = self.minucias_pixmap.scaled(
            int(self.minucias_pixmap.width() * self.zoom_factor),
            int(self.minucias_pixmap.height() * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.minutiae_points = {idx + 1: ponto for idx, ponto in enumerate(pontos)}
        cores_base, deltas_base = detect_singularities_trabalho(imagem_cv_original)
        self.pontos_core = cores_base[:2]
        self.pontos_delta = deltas_base[:1]
        self.limpar_correspondencias_singularidades()
        self.check_minucias.setEnabled(True)

        if hasattr(self, 'sobreposta_pixmap') and self.sobreposta_pixmap:
            qimg2 = self.sobreposta_pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w2, h2 = qimg2.width(), qimg2.height()
            ptr2 = qimg2.bits(); ptr2.setsize(qimg2.sizeInBytes())
            imagem_cv2 = np.array(ptr2, dtype=np.uint8).reshape((h2, w2, 4))
            escala = 0.35
            img2_red = cv2.resize(imagem_cv2, (0,0), fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
            _, pts2, direcoes_sobreposta = process_minutiae_trabalho(
                img2_red,
                cor=self.cor_minucias,
                raio=self.raio_minucias,
                quantidade=quantidade,
                distancia_threshold=distancia
            )
            self.direcoes_minucias_sobreposta = {}
            for idx in range(len(pts2)):
                if idx < len(direcoes_sobreposta):
                    self.direcoes_minucias_sobreposta[idx + 1] = direcoes_sobreposta[idx]
                else:
                    self.direcoes_minucias_sobreposta[idx + 1] = (0, -1)
            self.pontos_sobreposta = [(int(x/escala), int(y/escala), sc) for x, y, sc in pts2]
            self.minucias_detectadas_sobreposta = [(int(x/escala), int(y/escala)) for x, y, _ in pts2]
            cores_sobreposta, deltas_sobreposta = detect_singularities_trabalho(imagem_cv2)
            self.pontos_core_sobreposta_detectados = cores_sobreposta[:2]
            self.pontos_delta_sobreposta_detectados = deltas_sobreposta[:1]
             
        if hasattr(self, 'transparencia_dialog') and self.transparencia_dialog.isVisible():
            self.transparencia_dialog.atualizar_estado_minucias()
            self.transparencia_dialog.atualizar_estado_marcacao()
            self.transparencia_dialog.atualizar_estado_marcacao()

        if hasattr(self, 'transparencia_dialog'):
            self.mostrar_numeros_marcacao = True
            self.transparencia_dialog.check_numeros_marcacao.setEnabled(True)
            self.transparencia_dialog.check_numeros_marcacao.setChecked(True)
            self.settings.setValue("mostrar_numeros_marcacao", True)
            self.settings.sync()

        self.minucias_detectadas = True

        self.atualizar_status_lateral()
        self.update()

    def limpar_correspondencias_singularidades(self):
        if not hasattr(self, 'correspondencias'):
            self.correspondencias = {}

        for chave in list(self.correspondencias.keys()):
            try:
                chave_int = int(chave)
            except (TypeError, ValueError):
                continue
            if chave_int >= 1000:
                del self.correspondencias[chave]

        self.pontos_correspondentes_sobreposta = {
            tuple(valor) for valor in self.correspondencias.values()
        }

        self.registrar_log(
            "limpar_correspondencias_singularidades",
            {
                "cores_base": len(self.pontos_core),
                "deltas_base": len(self.pontos_delta),
            }
        )

    def redesenhar_minucias(self):
        """Redesenha as minúcias com a cor atual, sem recalcular nada."""
        if not hasattr(self, 'base_original') or not hasattr(self, 'minutiae_points'):
            return

        qimage = self.base_original.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        imagem_cv = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))

        imagem_minucias = np.zeros((height, width, 4), dtype=np.uint8)

        r, g, b = map(int, self.cor_minucias)
        for _, (x, y, _) in self.minutiae_points.items():
            cv2.circle(imagem_minucias, (x, y), self.raio_minucias, (b, g, r, 255), -1)

        qimage_minucias = QImage(imagem_minucias.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
        self.minucias_pixmap = QPixmap.fromImage(qimage_minucias)
        self.minucias_scaled = self.minucias_pixmap.scaled(
            int(self.minucias_pixmap.width() * self.zoom_factor),
            int(self.minucias_pixmap.height() * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.update()
    
    def obter_minucia_clicada(self, pos):
        """Retorna o índice da minúcia clicada, se houver, com base na posição do mouse."""
        if not hasattr(self, 'minutiae_points'):
            return None
        for idx, (x, y, _) in self.minutiae_points.items():
            draw_x = int(self.base_position_x + x * self.zoom_factor)
            draw_y = int(self.base_position_y + y * self.zoom_factor)
            if abs(pos.x() - draw_x) <= self.raio_minucias and abs(pos.y() - draw_y) <= self.raio_minucias:
                return idx
        return None

    def obter_pontos_snap_sobreposta(self, idx_base):
        if idx_base >= 2000:
            return list(getattr(self, "pontos_core_sobreposta_detectados", []))
        if 1000 <= idx_base < 2000:
            return list(getattr(self, "pontos_delta_sobreposta_detectados", []))
        return [(x, y) for x, y, _ in getattr(self, "pontos_sobreposta", [])]

    def obter_ponto_sobreposta_proximo(self, x_rel, y_rel, idx_base, limite=None):
        limite = self.raio_minucias if limite is None else limite
        ponto_mais_proximo = None
        menor_distancia = float("inf")

        for x2, y2 in self.obter_pontos_snap_sobreposta(idx_base):
            dist = math.hypot(x_rel - x2, y_rel - y2)
            if dist < menor_distancia and dist <= limite:
                menor_distancia = dist
                ponto_mais_proximo = (x2, y2)

        return ponto_mais_proximo

    def ajustar_margem_entre_imagens(self):
        """Ajusta a posição das imagens base e sobreposta mantendo uma margem entre elas."""
        MARGEM = int(5 * self.scale_x)

        if self.base_scaled and self.sobreposta_scaled:
            if self.base_position_x <= self.sobreposta_position_x:
                self.sobreposta_position_x = self.base_position_x + self.base_scaled.width() + MARGEM
                self.sobreposta_position_y = self.base_position_y
            else:
                self.base_position_x = self.sobreposta_position_x - self.base_scaled.width() - MARGEM
                self.base_position_y = self.sobreposta_position_y

    def salvar_cor_aplicada(self, cor, imagem):
        """Salva a cor aplicada na imagem base ou sobreposta e armazena a configuração no QSettings."""
        if imagem == "base":
            self.settings.setValue("cor_base", cor.name())
        elif imagem == "sobreposta":
            self.settings.setValue("cor_sobreposta", cor.name())
        self.settings.sync()

    def obter_cor_aplicada(self, pixmap):
        """Recupera a cor aplicada a um QPixmap, se possível. Como a cor foi aplicada diretamente à imagem, essa função é um placeholder."""
        if pixmap is None:
            return None

        if pixmap == self.base_colored:
            return self.cor_base_salva
        elif pixmap == self.sobreposta_colored:
            return self.cor_sobreposta_salva
        return None

    def carregar_imagem_base(self):
        """Abre um diálogo para selecionar e carregar uma imagem base, preparando-a para edição e alinhamento."""
        self.limpar_minucias_base()
        
        if self.base_colored:
            self.cor_base_salva = self.obter_cor_aplicada(self.base_colored)

        arquivo_imagem = escolher_arquivo(self, "Escolha uma imagem base", "Imagens (*.png)")

        if not arquivo_imagem:
            return

        self.arquivo_base = arquivo_imagem

        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.zoom_factor_global = 0.40 * min(self.scale_x, self.scale_y)

        imagem_cv = cv2.imread(arquivo_imagem, cv2.IMREAD_UNCHANGED)

        if imagem_cv is None:
            print("Erro ao carregar a imagem base.")
            return
        
        if self.cor_base_salva is None:
            imagem_cv[:, :, :3] = 0

        if imagem_cv.shape[2] == 4:
            altura, largura, _ = imagem_cv.shape
            qimage = QImage(imagem_cv.data, largura, altura, 4 * largura, QImage.Format.Format_RGBA8888)
        else:
            imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
            altura, largura, _ = imagem_rgb.shape
            qimage = QImage(imagem_rgb.data, largura, altura, 3 * largura, QImage.Format.Format_RGB888)

        self.base_pixmap = QPixmap.fromImage(qimage)

        if self.cor_base_salva:
            cor_dialog = CorDialog(self)
            self.base_colored = cor_dialog.aplicar_cor_na_imagem(self.base_pixmap.copy(), self.cor_base_salva)
            self.base_scaled = self.base_colored.scaled(
                int(self.base_pixmap.width() * self.zoom_factor_global),
                int(self.base_pixmap.height() * self.zoom_factor_global),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            self.base_colored = None
            self.base_scaled = self.base_pixmap.scaled(
                int(self.base_pixmap.width() * self.zoom_factor_global),
                int(self.base_pixmap.height() * self.zoom_factor_global),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if hasattr(self, 'cor_base_salva') and self.cor_base_salva:
            cor_dialog = CorDialog(self)
            self.base_colored = cor_dialog.aplicar_cor_na_imagem(self.base_pixmap.copy(), self.cor_base_salva)
        else:
            self.base_colored = None

        if self.cor_base_salva:
            cor_dialog = CorDialog(self)
            self.base_colored = cor_dialog.aplicar_cor_na_imagem(self.base_pixmap.copy(), self.cor_base_salva)

        self.base_scaled = self.base_pixmap.scaled(
            int(self.base_pixmap.width() * self.zoom_factor_global),
            int(self.base_pixmap.height() * self.zoom_factor_global),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        if self.sobreposta_colored:
            self.sobreposta_scaled = self.sobreposta_colored.copy()

        self.history.clear()
        self.current_index = -1
        self.update()
        self.update()

        dialog = SelecaoDialog(self, self.scale_x, self.scale_y)
        posicionar_dialogo_no_parent(dialog, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.nome_para_salvar = dialog.nome_selecionado

        if self.cor_base_salva:
            cor_dialog = CorDialog(self)
            self.base_colored = cor_dialog.aplicar_cor_na_imagem(self.base_pixmap.copy(), self.cor_base_salva)
        
        self.check_base_imagem.setEnabled(True)  
        self.alterar_transparencia_imagem_base()   
        self.ajustar_margem_entre_imagens()
        self.imagem_base_ok.emit(True) 
        self.settings.setValue("checkbox_base", True) 
        self.settings.sync()
        reaplicar_cor(self)
        self.check_base_imagem.setEnabled(True)
        self.sincronizar_imagens()
        self.update()
        self.update()  
        self.registrar_log("carregar_imagem_base", {"arquivo": arquivo_imagem})
        self.atualizar_status_lateral()
        self.imagem_base_carregada.emit()

    def carregar_imagem_sobreposta(self):
        """Abre um diálogo para carregar a imagem sobreposta e ajusta seus parâmetros iniciais."""         
        self.limpar_minucias_sobreposta()
        
        if self.sobreposta_colored:
            self.cor_sobreposta_salva = self.obter_cor_aplicada(self.sobreposta_colored)

        arquivo_imagem = escolher_arquivo(self, "Escolha uma imagem sobreposta", "Imagens (*.png)")

        if not arquivo_imagem:
            return

        self.arquivo_sobreposta = arquivo_imagem
        self.cor_sobreposta_alterada = False

        self.registrar_log("iniciar_carregamento_sobreposta", {"arquivo": arquivo_imagem})

        self.history.clear()
        self.current_index = -1

        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.zoom_factor_global = 0.40 * min(self.scale_x, self.scale_y)

        imagem_cv = cv2.imread(arquivo_imagem, cv2.IMREAD_UNCHANGED)

        if imagem_cv is None:
            print("Erro ao carregar a imagem sobreposta.")
            return
        
        if self.cor_sobreposta_salva is None:
            imagem_cv[:, :, :3] = 0

        if imagem_cv.shape[2] == 4:
            altura, largura, _ = imagem_cv.shape
            qimage = QImage(imagem_cv.data, largura, altura, 4 * largura, QImage.Format.Format_RGBA8888)
        else:
            imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
            altura, largura, _ = imagem_rgb.shape
            qimage = QImage(imagem_rgb.data, largura, altura, 3 * largura, QImage.Format.Format_RGB888)

        self.sobreposta_pixmap = QPixmap.fromImage(qimage)

        if self.cor_sobreposta_salva:
            cor_dialog = CorDialog(self)
            self.sobreposta_colored = cor_dialog.aplicar_cor_na_imagem(self.sobreposta_pixmap.copy(), self.cor_sobreposta_salva)
            self.sobreposta_scaled = self.sobreposta_colored.copy()

        if hasattr(self, 'cor_sobreposta_salva') and self.cor_sobreposta_salva:
            cor_dialog = CorDialog(self)
            self.sobreposta_colored = cor_dialog.aplicar_cor_na_imagem(self.sobreposta_pixmap.copy(), self.cor_sobreposta_salva)
        else:
            self.sobreposta_colored = None

        if self.sobreposta_colored:
            self.sobreposta_scaled = self.sobreposta_colored.copy()
        else:
            self.sobreposta_scaled = self.sobreposta_pixmap.scaled(
                int(self.sobreposta_pixmap.width() * self.zoom_factor_global),
                int(self.sobreposta_pixmap.height() * self.zoom_factor_global),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if self.sobreposta_pixmap:
            self.largura_original = self.sobreposta_pixmap.width()
            self.altura_original = self.sobreposta_pixmap.height()

        if self.base_colored:
            self.base_scaled = self.base_colored.copy()

        self.history.clear()
        self.current_index = -1
        self.update()
        self.update()

        if self.cor_sobreposta_salva:
            cor_dialog = CorDialog(self)
            self.sobreposta_colored = cor_dialog.aplicar_cor_na_imagem(self.sobreposta_pixmap.copy(), self.cor_sobreposta_salva)

        self.check_imagem_sobreposta.setEnabled(True) 
        self.alterar_transparencia_imagem_sobreposta()  
        self.ajustar_margem_entre_imagens()
        self.imagem_sobreposta_carregada.emit(True) 
        self.settings.setValue("checkbox_sobreposta", True)   
        self.settings.sync()
        reaplicar_cor(self)
        self.check_imagem_sobreposta.setEnabled(True)
        self.sincronizar_imagens()
        self.update()
        self.update()  
        self.sincronizar_imagens()
        self.ajustar_margem_entre_imagens()
        self.update()
        self.registrar_log("carregar_imagem_sobreposta", {"arquivo": arquivo_imagem})
        self.atualizar_status_lateral()

    def sincronizar_imagens(self):
        """Redimensiona e alinha as imagens base e sobreposta de acordo com os fatores de escala."""
        self.zoom_factor_global = 0.40 * min(self.scale_x, self.scale_y)

        if self.base_pixmap:
            if self.base_colored:
                self.base_scaled = self.base_colored.scaled(
                    int(self.base_pixmap.width() * self.zoom_factor_global),
                    int(self.base_pixmap.height() * self.zoom_factor_global),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                self.base_scaled = self.base_pixmap.scaled(
                    int(self.base_pixmap.width() * self.zoom_factor_global),
                    int(self.base_pixmap.height() * self.zoom_factor_global),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            
            if hasattr(self, 'minucias_pixmap') and self.minucias_pixmap:
                self.minucias_scaled = self.minucias_pixmap.scaled(
                    int(self.minucias_pixmap.width() * self.zoom_factor_global),
                    int(self.minucias_pixmap.height() * self.zoom_factor_global),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

        if self.sobreposta_pixmap:
            if self.sobreposta_colored:
                self.sobreposta_scaled = self.sobreposta_colored.scaled(
                    int(self.sobreposta_pixmap.width() * self.zoom_factor_global),
                    int(self.sobreposta_pixmap.height() * self.zoom_factor_global),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                self.sobreposta_scaled = self.sobreposta_pixmap.scaled(
                    int(self.sobreposta_pixmap.width() * self.zoom_factor_global),
                    int(self.sobreposta_pixmap.height() * self.zoom_factor_global),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

        self.update()

    def wheelEvent(self, event):
        mouse_pos = event.position().toPoint()
        zoom_factor_old = self.zoom_factor

        if event.angleDelta().y() > 0:
            self.zoom_factor = min(self.zoom_factor * 1.05, 0.55)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.05, 0.25)

        scale_factor = self.zoom_factor / zoom_factor_old

        if self.base_pixmap:
            pixmap_base = self.base_colored if self.base_colored else self.base_pixmap
            original_width = self.base_pixmap.width()
            original_height = self.base_pixmap.height()
            new_width = int(original_width * self.zoom_factor)
            new_height = int(original_height * self.zoom_factor)
            self.base_scaled = pixmap_base.scaled(
                new_width,
                new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.base_position_x = mouse_pos.x() + (self.base_position_x - mouse_pos.x()) * scale_factor
            self.base_position_y = mouse_pos.y() + (self.base_position_y - mouse_pos.y()) * scale_factor

        if self.sobreposta_pixmap:
            pixmap_sobre = self.sobreposta_colored if self.sobreposta_colored else self.sobreposta_pixmap
            original_width = self.sobreposta_pixmap.width()
            original_height = self.sobreposta_pixmap.height()
            new_width = int(original_width * self.zoom_factor)
            new_height = int(original_height * self.zoom_factor)
            self.sobreposta_scaled = pixmap_sobre.scaled(
                new_width,
                new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.sobreposta_position_x = mouse_pos.x() + (self.sobreposta_position_x - mouse_pos.x()) * scale_factor
            self.sobreposta_position_y = mouse_pos.y() + (self.sobreposta_position_y - mouse_pos.y()) * scale_factor

        if hasattr(self, 'minucias_pixmap') and self.minucias_pixmap:
            self.minucias_scaled = self.minucias_pixmap.scaled(
                int(self.minucias_pixmap.width() * self.zoom_factor),
                int(self.minucias_pixmap.height() * self.zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        self.ajustar_margem_entre_imagens()
        self.update()
        self.registrar_log(
            "zoom",
            {
                "zoom_anterior": float(zoom_factor_old),
                "zoom_atual": float(self.zoom_factor),
                "delta": int(event.angleDelta().y()),
            },
            event.position()
        )

    def paintEvent(self, event):
        """Desenha a interface gráfica, incluindo imagens e ajustes visuais como transparência e rotação."""
        painter = QPainter(self)

        largura = int(1920 * self.scale_x)
        altura = int(1080 * self.scale_y)
        x = int(((self.width() - (240 * self.scale_x)) - largura))
        y = int((self.height() - altura) * self.scale_y)

        painter.setBrush(self.rect_color)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawRect(x, y, largura, altura)

        region = QRegion(x, y, largura, altura)
        painter.setClipRegion(region)

        if self.easter_egg_active:
            painter.drawPixmap(self.base_position_x, self.base_position_y, self.easter_egg_image)
        else:
            if self.base_scaled:
                painter.setOpacity(self.transparencia if self.check_base_imagem.isChecked() else 1.0)
                base_x = self.base_position_x
                base_y = self.base_position_y
                painter.drawPixmap(int(base_x), int(base_y), self.base_scaled)

            if self.sobreposta_scaled:
                painter.setOpacity(self.transparencia if self.check_imagem_sobreposta.isChecked() else 1.0)
                painter.drawPixmap(int(self.sobreposta_position_x), int(self.sobreposta_position_y), self.sobreposta_scaled)
                
                if hasattr(self, 'pontos_sobreposta'):
                    raio2 = int(self.raio_minucias * self.zoom_factor)
                    for idx, (x2, y2, _) in enumerate(self.pontos_sobreposta, start=1):
                        px = int(self.sobreposta_position_x + x2 * self.zoom_factor)
                        py = int(self.sobreposta_position_y + y2 * self.zoom_factor)
                        
                        r, g, b = self.cor_minucias
                        cor = QColor(g,b,r)
                        numero = None

                        for i, (x2, y2) in enumerate(self.minucias_detectadas_sobreposta):
                            if (x2, y2) in self.pontos_correspondentes_sobreposta:
                                for id_base, ponto in self.correspondencias.items():
                                    if abs(x2 - ponto[0]) < 1 and abs(y2 - ponto[1]) < 1:
                                        numero = id_base
                                        break
                            else:
                                numero = None

                        painter.setOpacity(self.transparencia if self.check_minucias.isChecked() else 1.0)
                        painter.setBrush(cor)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(px - raio2, py - raio2, raio2 * 2, raio2 * 2)

                    painter.setOpacity(self.transparencia if self.check_minucias.isChecked() else 1.0)
                    for x_delta, y_delta in getattr(self, 'pontos_delta_sobreposta_detectados', []):
                        px = int(self.sobreposta_position_x + x_delta * self.zoom_factor)
                        py = int(self.sobreposta_position_y + y_delta * self.zoom_factor)
                        painter.setBrush(self.cor_delta)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(px - raio2, py - raio2, raio2 * 2, raio2 * 2)

                    for x_core, y_core in getattr(self, 'pontos_core_sobreposta_detectados', []):
                        px = int(self.sobreposta_position_x + x_core * self.zoom_factor)
                        py = int(self.sobreposta_position_y + y_core * self.zoom_factor)
                        painter.setBrush(self.cor_core)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(px - raio2, py - raio2, raio2 * 2, raio2 * 2)

                    if self.mostrar_direcao_minucias:
                        for idx, (x2, y2, _) in enumerate(self.pontos_sobreposta, start=1):
                            start_x = int(self.sobreposta_position_x + x2 * self.zoom_factor)
                            start_y = int(self.sobreposta_position_y + y2 * self.zoom_factor)

                            if hasattr(self, 'direcoes_minucias_sobreposta') and idx in self.direcoes_minucias_sobreposta:
                                dx, dy = self.direcoes_minucias_sobreposta[idx]
                                end_x = int(start_x + dx * self.tamanho_direcao_altura * self.zoom_factor)
                                end_y = int(start_y + dy * self.tamanho_direcao_altura * self.zoom_factor)

                                pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                                painter.setPen(pen)
                                painter.drawLine(start_x, start_y, end_x, end_y)

            if hasattr(self, 'minucias_scaled') and self.minucias_scaled:
                painter.setOpacity(self.transparencia if self.check_minucias.isChecked() else 1.0)
                painter.drawPixmap(int(self.base_position_x), int(self.base_position_y), self.minucias_scaled)

                if hasattr(self, 'minutiae_points'):
                    font = QFont("Arial", 26, QFont.Weight.Bold)
                    painter.setFont(font)

                    if self.mostrar_numeros_marcacao:
                        for idx, (x, y, _) in self.minutiae_points.items():
                            draw_x = int(self.base_position_x + x * self.zoom_factor)
                            draw_y = int(self.base_position_y + y * self.zoom_factor)

                            if self.minucia_selecionada == idx:
                                raio_com_escala = int(self.raio_minucias * self.zoom_factor)
                                painter.setBrush(Qt.BrushStyle.NoBrush)
                                r, g, b = self.cor_minucias
                                cor_borda = QColor(g, r, b)
                                painter.setPen(QPen(cor_borda, 3))
                                x_pos = self.base_position_x + x * self.zoom_factor
                                y_pos = self.base_position_y + y * self.zoom_factor

                                painter.drawEllipse(
                                    int(x_pos - raio_com_escala),
                                    int(y_pos - raio_com_escala),
                                    raio_com_escala * 2,
                                    raio_com_escala * 2
                                )
                            painter.setPen(QColor(0, 0, 0))
                            painter.drawText(draw_x + 11, draw_y + 11, str(idx))
                            painter.setPen(self.cor_marcacao)
                            painter.drawText(draw_x + 10, draw_y + 10, str(idx))
                    
                    if self.mostrar_direcao_minucias:
                        for idx, (x, y, _) in self.minutiae_points.items():
                            start_x = int(self.base_position_x + x * self.zoom_factor)
                            start_y = int(self.base_position_y + y * self.zoom_factor)

                            if hasattr(self, 'direcoes_minucias_base') and idx in self.direcoes_minucias_base:
                                dx, dy = self.direcoes_minucias_base[idx]
                                end_x = int(start_x + dx * self.tamanho_direcao_altura * self.zoom_factor)
                                end_y = int(start_y + dy * self.tamanho_direcao_altura * self.zoom_factor)

                                pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                                painter.setPen(pen)
                                painter.drawLine(start_x, start_y, end_x, end_y)

            raio = int(self.raio_minucias * self.zoom_factor)

            for idx, (x, y) in enumerate(self.pontos_delta, start=1):
                px = int(self.base_position_x + x * self.zoom_factor)
                py = int(self.base_position_y + y * self.zoom_factor)
                num = 1000 + idx
                raio = int(self.raio_minucias * self.zoom_factor)

                r, g, b = self.cor_minucias
                cor_bolinha = self.cor_delta
                cor_borda = QColor(*inverter_cor((self.cor_delta.red(), self.cor_delta.green(), self.cor_delta.blue())))

                painter.setBrush(cor_bolinha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(px - raio, py - raio, raio * 2, raio * 2)

                if self.minucia_selecionada == num:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(cor_borda, 3))
                    painter.drawEllipse(px - raio, py - raio, raio * 2, raio * 2)

                if self.mostrar_numeros_marcacao:
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(px + 11, py + 11, str(idx))
                    r, g, b = self.cor_delta.red(), self.cor_delta.green(), self.cor_delta.blue()
                    painter.setPen(QColor(b, r, g)) 
                    painter.drawText(px + 10, py + 10, str(idx))

            for idx, (x, y) in enumerate(self.pontos_core, start=1):
                px = int(self.base_position_x + x * self.zoom_factor)
                py = int(self.base_position_y + y * self.zoom_factor)
                num = 2000 + idx
                raio = int(self.raio_minucias * self.zoom_factor)

                r, g, b = self.cor_minucias
                cor_bolinha = self.cor_core
                cor_borda = QColor(*inverter_cor((self.cor_core.red(), self.cor_core.green(), self.cor_core.blue())))

                painter.setBrush(cor_bolinha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(px - raio, py - raio, raio * 2, raio * 2)

                if self.minucia_selecionada == num:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(cor_borda, 3))
                    painter.drawEllipse(px - raio, py - raio, raio * 2, raio * 2)

                if self.mostrar_numeros_marcacao:
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(px + 11, py + 11, str(idx))
                    r, g, b = self.cor_core.red(), self.cor_core.green(), self.cor_core.blue()
                    painter.setPen(QColor(b, r, g)) 
                    painter.drawText(px + 10, py + 10, str(idx))

            offset = len(self.minutiae_points) if hasattr(self, "minutiae_points") else 0
            raio_com_escala = int(self.raio_minucias * self.zoom_factor)
            r, g, b = self.cor_minucias
            cor_bolinha = QColor(r, g, b)
            cor_borda = QColor(g, r, b)
            cor_texto = self.cor_marcacao

            for idx, (x, y) in enumerate(self.pontos_minucia_manual, start=1):
                px = int(self.base_position_x + x * self.zoom_factor)
                py = int(self.base_position_y + y * self.zoom_factor)
                num = offset + idx

                b, g, r = self.cor_minucias
                painter.setBrush(QColor(r, g, b))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(px - raio_com_escala, py - raio_com_escala, raio_com_escala * 2, raio_com_escala * 2)

                if self.minucia_selecionada == num:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(cor_borda, 3))
                    painter.drawEllipse(
                        px - raio_com_escala,
                        py - raio_com_escala,
                        raio_com_escala * 2,
                        raio_com_escala * 2
                    )

                if self.mostrar_numeros_marcacao:
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(px + 11, py + 11, str(num))
                    painter.setPen(self.cor_marcacao)
                    painter.drawText(px + 10, py + 10, str(num))

                if self.mostrar_direcao_minucias and hasattr(self, 'direcoes_minucias_base') and getattr(self, 'minutiae_points', None):
                    min_dist = float('inf')
                    nearest_idx = None
                    for auto_idx, (x_auto, y_auto, _) in self.minutiae_points.items():
                        dist = math.hypot(x - x_auto, y - y_auto)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = auto_idx
                   
                    if nearest_idx in self.direcoes_minucias_base:
                        dx, dy = self.direcoes_minucias_base[nearest_idx]
                        start_x, start_y = px, py
                        end_x = int(start_x + dx * self.tamanho_direcao_altura * self.zoom_factor)
                        end_y = int(start_y + dy * self.tamanho_direcao_altura * self.zoom_factor)
                        pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                        painter.setPen(pen)
                        painter.drawLine(start_x, start_y, end_x, end_y)

        if hasattr(self, 'correspondencias'):
            for idx, pos in self.correspondencias.items():
                if idx >= 2000:
                    numero_puro = idx - 2000
                    cor_base = self.cor_core
                    cor_bolinha = QColor(*inverter_cor((cor_base.red(), cor_base.green(), cor_base.blue())))
                    cor_texto = QColor(cor_bolinha.blue(), cor_bolinha.red(), cor_bolinha.green()) 
                elif idx >= 1000:
                    numero_puro = idx - 1000
                    cor_base = self.cor_delta
                    cor_bolinha = QColor(*inverter_cor((cor_base.red(), cor_base.green(), cor_base.blue())))
                    cor_texto = QColor(cor_bolinha.blue(), cor_bolinha.red(), cor_bolinha.green()) 
                else:
                    numero_puro = idx if idx < 1000 else idx - 1000 if idx < 2000 else idx - 2000
                    r, g, b = self.cor_minucias
                    cor_bolinha = QColor(r, g, b)
                    r_m = self.cor_marcacao.red()
                    g_m = self.cor_marcacao.green()
                    b_m = self.cor_marcacao.blue()
                    cor_texto = QColor(b_m, r_m, g_m)

                raio_com_escala = int(self.raio_minucias * self.zoom_factor)
                x = self.sobreposta_position_x + pos[0] * self.zoom_factor
                y = self.sobreposta_position_y + pos[1] * self.zoom_factor

                if self.minucia_selecionada == idx:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    if idx >= 2000:
                        cor_base = self.cor_core
                    elif idx >= 1000:
                        cor_base = self.cor_delta
                    else:
                        r, g, b = self.cor_minucias
                        cor_base = QColor(r, g, b)
                    cor_borda = QColor(*inverter_cor((cor_base.red(), cor_base.green(), cor_base.blue())))
                    painter.setPen(QPen(cor_borda, 3))
                else:
                    painter.setBrush(cor_bolinha)
                    painter.setPen(Qt.PenStyle.NoPen)

                painter.drawEllipse(
                    int(x - raio_com_escala),
                    int(y - raio_com_escala),
                    2 * raio_com_escala,
                    2 * raio_com_escala
                )

                if self.mostrar_numeros_marcacao:
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(int(x) + 11, int(y) + 11, str(numero_puro))
                    painter.setPen(cor_texto)
                    painter.drawText(int(x) + 10, int(y) + 10, str(numero_puro))
                
                if self.mostrar_direcao_minucias:

                    if idx in self.direcoes_minucias_sobreposta:
                        dx, dy = self.direcoes_minucias_sobreposta[idx]

                    elif idx >= 1000:
                        offset_base = len(self.minutiae_points)
                        idx_manual = idx - 1000 - offset_base

                        if 0 <= idx_manual < len(self.pontos_minucia_manual):
                            ponto_manual = self.pontos_minucia_manual[idx_manual]
                            nearest_idx = min(
                                self.minutiae_points.keys(),
                                key=lambda i: math.hypot(
                                    ponto_manual[0] - self.minutiae_points[i][0],
                                    ponto_manual[1] - self.minutiae_points[i][1]
                                )
                            )
                            dx, dy = self.direcoes_minucias_base.get(nearest_idx, (0, 0))

                        elif 0 <= idx_manual < len(self.correspondencias):
                            ponto_corr = self.correspondencias.get(idx, None)
                            if ponto_corr:
                                if (
                                    getattr(self, 'pontos_sobreposta', [])
                                    and hasattr(self, 'direcoes_minucias_sobreposta')
                                ):
                                    nearest_idx = min(
                                        self.direcoes_minucias_sobreposta.keys(),
                                        key=lambda i: math.hypot(
                                            ponto_corr[0] - self.pontos_sobreposta[i - 1][0],
                                            ponto_corr[1] - self.pontos_sobreposta[i - 1][1]
                                        )
                                    )
                                    dx, dy = self.direcoes_minucias_sobreposta.get(nearest_idx, (0, 0))
                                else:
                                    dx, dy = (0, 0)
                            else:
                                dx, dy = (0, 0)
                        else:
                            dx, dy = (0, 0)

                    if dx != 0 or dy != 0:
                        pos_chave = (round(pos[0]), round(pos[1]))  
                        direcao_ja_existe = any(
                            math.hypot(pos_chave[0] - p[0], pos_chave[1] - p[1]) < 1
                            for p in self.minucias_detectadas_sobreposta
                        )

                        if not direcao_ja_existe:
                            start_x = int(self.sobreposta_position_x + pos[0] * self.zoom_factor)
                            start_y = int(self.sobreposta_position_y + pos[1] * self.zoom_factor)
                            end_x = int(start_x + dx * self.tamanho_direcao_altura * self.zoom_factor)
                            end_y = int(start_y + dy * self.tamanho_direcao_altura * self.zoom_factor)
                            pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                            painter.setPen(pen)
                            painter.drawLine(start_x, start_y, end_x, end_y)

        painter.setOpacity(1.0)

        if hasattr(self, "grafo_base") and hasattr(self, "grafo_dialog") and self.grafo_dialog and hasattr(self.grafo_dialog, "check_grafo") and self.grafo_dialog.check_grafo.isChecked():
            pen = QPen(self.cor_grafo, self.espessura_grafo) 
            pen.setColor(QColor(self.cor_grafo.red(), self.cor_grafo.green(), self.cor_grafo.blue(), int(self.transparencia_grafo * 255)))
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in self.grafo_base:
                painter.drawLine(
                    int(self.base_position_x + x1 * self.zoom_factor),
                    int(self.base_position_y + y1 * self.zoom_factor),
                    int(self.base_position_x + x2 * self.zoom_factor),
                    int(self.base_position_y + y2 * self.zoom_factor),
                )

        if hasattr(self, "grafo_sobreposta") and hasattr(self, "grafo_dialog") and self.grafo_dialog and hasattr(self.grafo_dialog, "check_grafo") and self.grafo_dialog.check_grafo.isChecked():
            pen = QPen(self.cor_grafo, self.espessura_grafo)
            pen.setColor(QColor(self.cor_grafo.red(), self.cor_grafo.green(), self.cor_grafo.blue(), int(self.transparencia_grafo * 255)))
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in self.grafo_sobreposta:
                painter.drawLine(
                    int(self.sobreposta_position_x + x1 * self.zoom_factor),
                    int(self.sobreposta_position_y + y1 * self.zoom_factor),
                    int(self.sobreposta_position_x + x2 * self.zoom_factor),
                    int(self.sobreposta_position_y + y2 * self.zoom_factor),
                )
            
        if hasattr(self, "grafo_matching") and hasattr(self, "matching_dialog") and self.matching_dialog and hasattr(self.matching_dialog, "check_matching") and self.matching_dialog.check_matching.isChecked():
            pen = QPen(self.cor_matching, self.espessura_matching)
            pen.setColor(QColor(self.cor_matching.red(), self.cor_matching.green(), self.cor_matching.blue(), int(self.transparencia_grafo * 255)))
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in self.grafo_matching:
                painter.drawLine(
                    int(self.base_position_x + x1 * self.zoom_factor),
                    int(self.base_position_y + y1 * self.zoom_factor),
                    int(self.sobreposta_position_x + x2 * self.zoom_factor),
                    int(self.sobreposta_position_y + y2 * self.zoom_factor),
                )

    def mousePressEvent(self, event):
        """Captura os eventos de clique do mouse para ações como seleção ou marcação de correspondência."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_position = event.position()

            if event.button() == Qt.MouseButton.LeftButton:
                if self.modo_marcacao in ["marcar_core", "marcar_delta", "marcar_minucia"]:
                    if not getattr(self, "minucias_detectadas", False):
                        QMessageBox.warning(self, "Aviso", "Você precisa primeiro detectar os pontos automáticos antes de marcar manualmente.")
                        return

            if self.modo_marcacao:
                acao, tipo = self.modo_marcacao.split("_")
                x_click = (event.position().x() - self.base_position_x) / self.zoom_factor
                y_click = (event.position().y() - self.base_position_y) / self.zoom_factor

                ponto = (int(x_click), int(y_click))
                self.editando_automatica = (acao == "editar" and hasattr(self, "minutiae_points"))
                
                if tipo == "minucia":
                    lista = self.pontos_minucia_manual
                    self.editando_automatica = False

                elif tipo == "delta":
                    lista = self.pontos_delta
                elif tipo == "core":
                    lista = self.pontos_core
                else:
                    return

                if acao == "marcar":
                    if tipo == "delta" and len(self.pontos_delta) >= 1:
                        return
                    if tipo == "core" and len(self.pontos_core) >= 2:
                        return

                    lista.append(ponto)
                    self.registrar_log(f"marcar_{tipo}", {"ponto": {"x": ponto[0], "y": ponto[1]}})
                    self.ponto_editando = None
                    self.minucia_selecionada = None  
                    self.update()
                    return

                elif acao == "editar":
                    for idx, (x, y) in enumerate(lista):
                        if abs(x_click - x) <= 15 and abs(y_click - y) <= 15:
                            self.ponto_editando = (tipo, idx)
                            break

                    if hasattr(self, "correspondencias"):
                        for idx, pos in self.correspondencias.items():
                            px, py = pos
                            px = self.sobreposta_position_x + px * self.zoom_factor
                            py = self.sobreposta_position_y + py * self.zoom_factor
                            if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                self.ponto_editando = ("correspondencia", idx)
                                break

                elif acao == "apagar":
                    apagou = False

                    if hasattr(self, "correspondencias"):
                        for idx, pos in list(self.correspondencias.items()):
                            if tipo == "minucia" and idx < 1000:
                                px = self.sobreposta_position_x + pos[0] * self.zoom_factor
                                py = self.sobreposta_position_y + pos[1] * self.zoom_factor
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    del self.correspondencias[idx]
                                    apagou = True
                                    break
                            elif tipo == "delta" and 1000 <= idx < 2000:
                                px = self.sobreposta_position_x + pos[0] * self.zoom_factor
                                py = self.sobreposta_position_y + pos[1] * self.zoom_factor
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    del self.correspondencias[idx]
                                    apagou = True
                                    break
                            elif tipo == "core" and idx >= 2000:
                                px = self.sobreposta_position_x + pos[0] * self.zoom_factor
                                py = self.sobreposta_position_y + pos[1] * self.zoom_factor
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    numero_core = idx
                                    indice_lista = numero_core - 2000 - 1
                                    if hasattr(self, 'correspondencias'):
                                        if numero_core in self.correspondencias:
                                            del self.correspondencias[numero_core]
                                    if 0 <= indice_lista < len(self.pontos_core):
                                        del self.pontos_core[indice_lista]
                                    apagou = True
                                    break

                    if not apagou:
                        if tipo == "core":
                            for idx, (x, y) in enumerate(self.pontos_core):
                                px = int(self.base_position_x + x * self.zoom_factor)
                                py = int(self.base_position_y + y * self.zoom_factor)
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    if hasattr(self, 'correspondencias'):
                                        num_corresp = 2000 + idx + 1
                                        if num_corresp in self.correspondencias:
                                            del self.correspondencias[num_corresp]
                                    del self.pontos_core[idx]
                                    apagou = True
                                    break

                        elif tipo == "delta":
                            for idx, (x, y) in enumerate(self.pontos_delta):
                                px = int(self.base_position_x + x * self.zoom_factor)
                                py = int(self.base_position_y + y * self.zoom_factor)
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    del self.pontos_delta[idx]
                                    if hasattr(self, 'correspondencias'):
                                        num_corresp = 1000 + idx + 1
                                        if num_corresp in self.correspondencias:
                                            del self.correspondencias[num_corresp]
                                    apagou = True
                                    break

                        elif tipo == "minucia":
                            offset = len(self.minutiae_points) if hasattr(self, "minutiae_points") else 0
                            for idx, (x, y) in enumerate(self.pontos_minucia_manual):
                                numero_global = offset + idx + 1
                                px = int(self.base_position_x + x * self.zoom_factor)
                                py = int(self.base_position_y + y * self.zoom_factor)
                                if numero_global > 20 and abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    del self.pontos_minucia_manual[idx]
                                    if hasattr(self, 'correspondencias'):
                                        if numero_global in self.correspondencias:
                                            del self.correspondencias[numero_global]
                                    apagou = True
                                    break

                            if not apagou:
                                for idx, (x, y) in enumerate(self.pontos_minucia_manual):
                                    px = int(self.base_position_x + x * self.zoom_factor)
                                    py = int(self.base_position_y + y * self.zoom_factor)
                                    if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                        del self.pontos_minucia_manual[idx]
                                        apagou = True
                                        break

                            if hasattr(self, "minutiae_points"):
                                for idx, (x, y, _) in self.minutiae_points.items():
                                    px = int(self.base_position_x + x * self.zoom_factor)
                                    py = int(self.base_position_y + y * self.zoom_factor)
                                    if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                        if idx in self.minutiae_points:
                                            if idx in self.minutiae_points:
                                                del self.minutiae_points[idx]

                                        if hasattr(self, "correspondencias") and idx + 1 in self.correspondencias:
                                            ponto_corresp = self.correspondencias[idx + 1]
                                            if ponto_corresp in self.pontos_correspondentes_sobreposta:
                                                self.pontos_correspondentes_sobreposta.remove(ponto_corresp)
                                            del self.correspondencias[idx + 1]

                                            if hasattr(self, "pontos_sobreposta"):
                                                self.pontos_sobreposta = [
                                                    p for p in self.pontos_sobreposta if (int(p[0]), int(p[1])) != (int(ponto_corresp[0]), int(ponto_corresp[1]))
                                                ]
                                            if hasattr(self, "minucias_detectadas_sobreposta"):
                                                self.minucias_detectadas_sobreposta = [
                                                    p for p in self.minucias_detectadas_sobreposta if (int(p[0]), int(p[1])) != (int(ponto_corresp[0]), int(ponto_corresp[1]))
                                                ]

                                        self.redesenhar_minucias()
                                        self.minucia_selecionada = None
                                        self.update()
                                        return
                            
                        if hasattr(self, "pontos_sobreposta"):
                            for i, (x, y, _) in enumerate(self.pontos_sobreposta):
                                px = int(self.sobreposta_position_x + x * self.zoom_factor)
                                py = int(self.sobreposta_position_y + y * self.zoom_factor)
                                if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                                    del self.pontos_sobreposta[i]
                                    if hasattr(self, "minucias_detectadas_sobreposta"):
                                        if i < len(self.minucias_detectadas_sobreposta):
                                            del self.minucias_detectadas_sobreposta[i]
                                    self.update()
                                    return

                    if apagou:
                        self.registrar_log(f"apagar_{tipo}")
                        self.minucia_selecionada = None
                        self.update()
                        return

                self.update()

            if self.base_scaled:
                idx = self.obter_minucia_clicada(event.pos())
                if idx is not None:
                    self.minucia_selecionada = idx
                    self.update()
                    return
                
            if hasattr(self, "pontos_minucia_manual"):
                for idx, (x, y) in enumerate(self.pontos_minucia_manual, start=1):
                    px = int(self.base_position_x + x * self.zoom_factor)
                    py = int(self.base_position_y + y * self.zoom_factor)
                    if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                        offset = len(self.minutiae_points) if hasattr(self, "minutiae_points") else 0
                        self.minucia_selecionada = offset + idx
                        self.update()
                        return
        
            if hasattr(self, "pontos_delta"):
                for idx, (x, y) in enumerate(self.pontos_delta, start=1):
                    px = int(self.base_position_x + x * self.zoom_factor)
                    py = int(self.base_position_y + y * self.zoom_factor)
                    if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                        self.minucia_selecionada = 1000 + idx  
                        self.update()
                        return
                    
            if hasattr(self, "pontos_core"):
                for idx, (x, y) in enumerate(self.pontos_core, start=1):
                    px = int(self.base_position_x + x * self.zoom_factor)
                    py = int(self.base_position_y + y * self.zoom_factor)
                    if abs(event.position().x() - px) <= 15 and abs(event.position().y() - py) <= 15:
                        self.minucia_selecionada = 2000 + idx 
                        self.update()
                        return

            if self.minucia_selecionada is not None:
                idx_base = self.minucia_selecionada
                if not hasattr(self, 'correspondencias'):
                    self.correspondencias = {}

                click_x = (event.position().x() - self.sobreposta_position_x) / self.zoom_factor
                click_y = (event.position().y() - self.sobreposta_position_y) / self.zoom_factor

                ponto_mais_proximo = self.obter_ponto_sobreposta_proximo(
                    click_x,
                    click_y,
                    idx_base,
                )

                if ponto_mais_proximo is not None:
                    for ponto in list(self.pontos_correspondentes_sobreposta):
                        if any(abs(ponto[0] - p[0]) < 1 and abs(ponto[1] - p[1]) < 1 for p in self.correspondencias.values() if p != ponto):
                            self.pontos_correspondentes_sobreposta.discard(ponto)
                    self.correspondencias[self.minucia_selecionada] = ponto_mais_proximo
                    self.pontos_correspondentes_sobreposta.add(ponto_mais_proximo)

                else:
                    self.correspondencias[self.minucia_selecionada] = (
                        (event.position().x() - self.sobreposta_position_x) / self.zoom_factor,
                        (event.position().y() - self.sobreposta_position_y) / self.zoom_factor
                    )
                    self.pontos_correspondentes_sobreposta.add(self.correspondencias[self.minucia_selecionada])

                self.registrar_log(
                    "marcar_correspondencia",
                    {
                        "id_base": idx_base,
                        "ponto_sobreposta": {
                            "x": self.correspondencias[idx_base][0],
                            "y": self.correspondencias[idx_base][1],
                        },
                    }
                )
                self.minucia_selecionada = None
                self.update()
                return

    def mouseReleaseEvent(self, event):
        """Libera os eventos de arrastar e rotação ao soltar o botão do mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.ponto_editando = None
            self.dragging = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.rotating = False

    def mouseMoveEvent(self, event):
        if self.ponto_editando:
            tipo, idx = self.ponto_editando

            if tipo == "correspondencia":
                if idx < 1000 and self.modo_marcacao != "editar_minucia":
                    return
                
                if 1000 <= idx < 2000 and self.modo_marcacao != "editar_delta":
                    return
                
                if idx >= 2000 and self.modo_marcacao != "editar_core":
                    return

                if hasattr(self, "correspondencias") and idx in self.correspondencias:

                        x_rel = (event.position().x() - self.sobreposta_position_x) / self.zoom_factor
                        y_rel = (event.position().y() - self.sobreposta_position_y) / self.zoom_factor

                      
                        threshold_px = 8
                        threshold_rel = threshold_px / self.zoom_factor
                        ponto_mais_proximo = self.obter_ponto_sobreposta_proximo(
                            x_rel,
                            y_rel,
                            idx,
                            threshold_rel,
                        )
                        if ponto_mais_proximo is not None:
                            x_rel, y_rel = ponto_mais_proximo

                        self.correspondencias[idx] = (x_rel, y_rel)
                        self.update()
                        return

            x_click = (event.position().x() - self.base_position_x) / self.zoom_factor
            y_click = (event.position().y() - self.base_position_y) / self.zoom_factor

            if tipo == "minucia":
                if idx < len(self.pontos_minucia_manual):
                    self.pontos_minucia_manual[idx] = (int(x_click), int(y_click))
            elif tipo == "delta":
                if idx < len(self.pontos_delta):
                    self.pontos_delta[idx] = (int(x_click), int(y_click))
            elif tipo == "core":
                if idx < len(self.pontos_core):
                    self.pontos_core[idx] = (int(x_click), int(y_click))

            self.update()

    def closeEvent(self, event):
        """Garante que as configurações só sejam apagadas se o fechamento for intencional."""
        
        self.settings.setValue("checkbox_base", False)
        self.settings.setValue("checkbox_sobreposta", False)
        self.settings.sync()

        event.accept()

    def alterar_transparencia_imagem_base(self):
        """Ativa ou desativa a transparência da imagem base."""
        if self.base_pixmap is None or self.base_pixmap.isNull():
            self.check_base_imagem.setEnabled(False)
            self.check_base_imagem.setChecked(False)
            QMessageBox.warning(self, "Atenção", "A imagem base não está carregada.")
        else:
            self.check_base_imagem.setEnabled(True)
            self.check_base_imagem.setVisible(True)  

            if self.base_pixmap:
                scaled_width = int(self.base_pixmap.width() * self.zoom_factor)
                scaled_height = int(self.base_pixmap.height() * self.zoom_factor)

                self.base_scaled = self.base_pixmap.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                if self.base_colored is not None:
                    self.base_scaled = self.base_colored.scaled(
                        scaled_width,
                        scaled_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )

                if self.check_base_imagem.isChecked():
                    base_with_transparency = self.base_scaled.copy()
                    painter = QPainter(base_with_transparency)
                    painter.setOpacity(self.transparencia)
                    painter.end()
                    self.base_scaled = base_with_transparency

                self.update()

    def alterar_transparencia_imagem_sobreposta(self):
        """Ativa ou desativa a transparência da imagem sobreposta."""
        if self.sobreposta_pixmap is None or self.sobreposta_pixmap.isNull():
            self.check_imagem_sobreposta.setEnabled(False)
            self.check_imagem_sobreposta.setChecked(False)
            QMessageBox.warning(self, "Atenção", "A imagem sobreposta não está carregada.")
        else:
            self.check_imagem_sobreposta.setEnabled(True)
            self.check_imagem_sobreposta.setVisible(True)  

            if self.sobreposta_pixmap:
                scaled_width = int(self.sobreposta_pixmap.width() * self.zoom_factor)
                scaled_height = int(self.sobreposta_pixmap.height() * self.zoom_factor)

                self.sobreposta_scaled = self.sobreposta_pixmap.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                if self.sobreposta_colored is not None:
                    self.sobreposta_scaled = self.sobreposta_colored.scaled(
                        scaled_width,
                        scaled_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )

                if self.check_imagem_sobreposta.isChecked():
                    sobreposta_with_transparency = self.sobreposta_scaled.copy()
                    painter = QPainter(sobreposta_with_transparency)
                    painter.setOpacity(self.transparencia)
                    painter.end()
                    self.sobreposta_scaled = sobreposta_with_transparency

                self.update() 

    def alterar_transparencia_minucias(self):
        """Ativa ou desativa a transparência das minúcias."""
        if not hasattr(self, 'minucias_pixmap') or self.minucias_pixmap is None:
            self.check_minucias.setEnabled(False)
            self.check_minucias.setChecked(False)
            QMessageBox.warning(self, "Atenção", "As minúcias não foram detectadas.")
        else:
            self.check_minucias.setEnabled(True)
            self.check_minucias.setVisible(True)
            self.update()

    def resetar_zoom(self):
        """Reseta o fator de zoom, centraliza as imagens, mantém as cores e alinha tamanhos."""
        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.base_position_x = int(10 * self.scale_x)
        self.base_position_y = int(115 * self.scale_y)

        base_pixmap_to_use = self.base_colored if self.base_colored else self.base_pixmap
        sobreposta_pixmap_to_use = self.sobreposta_colored if self.sobreposta_colored else self.sobreposta_pixmap

        if base_pixmap_to_use:
            self.base_scaled = base_pixmap_to_use.scaled(
                int(base_pixmap_to_use.width() * self.zoom_factor),
                int(base_pixmap_to_use.height() * self.zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if sobreposta_pixmap_to_use and self.base_scaled:
            altura_base = self.base_scaled.height()
            self.sobreposta_scaled = sobreposta_pixmap_to_use.scaledToHeight(
                altura_base,
                Qt.TransformationMode.SmoothTransformation
            )

        if hasattr(self, 'minucias_pixmap') and self.minucias_pixmap:
            self.minucias_scaled = self.minucias_pixmap.scaled(
                int(self.minucias_pixmap.width() * self.zoom_factor),
                int(self.minucias_pixmap.height() * self.zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if self.base_scaled and self.sobreposta_scaled:
            self.sobreposta_position_x = int(self.base_position_x + self.base_scaled.width() + 30)
            self.sobreposta_position_y = self.base_position_y

        self.ajustar_margem_entre_imagens()
        self.update()
        self.registrar_log("resetar_zoom")

    def obter_estado_log(self):
        """Monta um retrato serializavel do estado visual atual."""
        return {
            "zoom_factor": float(self.zoom_factor),
            "base": {
                "carregada": self.base_pixmap is not None,
                "arquivo": getattr(self, "arquivo_base", None),
                "x": float(self.base_position_x),
                "y": float(self.base_position_y),
                "largura": self.base_scaled.width() if self.base_scaled else None,
                "altura": self.base_scaled.height() if self.base_scaled else None,
            },
            "sobreposta": {
                "carregada": self.sobreposta_pixmap is not None,
                "arquivo": getattr(self, "arquivo_sobreposta", None),
                "x": float(self.sobreposta_position_x),
                "y": float(self.sobreposta_position_y),
                "largura": self.sobreposta_scaled.width() if self.sobreposta_scaled else None,
                "altura": self.sobreposta_scaled.height() if self.sobreposta_scaled else None,
            },
            "marcacao": {
                "modo": self.modo_marcacao,
                "minucia_selecionada": self.minucia_selecionada,
                "minucias_auto": len(getattr(self, "minutiae_points", {})),
                "minucias_manuais": len(getattr(self, "pontos_minucia_manual", [])),
                "deltas": len(getattr(self, "pontos_delta", [])),
                "cores": len(getattr(self, "pontos_core", [])),
                "correspondencias": len(getattr(self, "correspondencias", {})),
            },
            "opcoes": {
                "transparencia": float(self.transparencia),
                "base_transparente": self.check_base_imagem.isChecked(),
                "sobreposta_transparente": self.check_imagem_sobreposta.isChecked(),
                "minucias_transparentes": self.check_minucias.isChecked(),
                "mostrar_direcao_minucias": bool(self.mostrar_direcao_minucias),
            },
        }

    def montar_dados_log(self):
        """Retorna um snapshot completo do trabalho para restauracao posterior."""
        dados = {
            "versao": 2,
            "tipo": "benah_estado_trabalho",
            "sessao": self.log_session_id,
            "inicio": self.log_inicio,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "imagens": {
                "base": {
                    "arquivo_original": getattr(self, "arquivo_base", None),
                    "arquivo_zip": "imagem_base_original.png",
                },
                "sobreposta": {
                    "arquivo_original": getattr(self, "arquivo_sobreposta", None),
                    "arquivo_zip": "imagem_sobreposta_original.png",
                },
            },
            "visual": {
                "zoom_factor": float(self.zoom_factor),
                "base_position_x": float(self.base_position_x),
                "base_position_y": float(self.base_position_y),
                "sobreposta_position_x": float(self.sobreposta_position_x),
                "sobreposta_position_y": float(self.sobreposta_position_y),
                "transparencia": float(self.transparencia),
                "base_transparente": self.check_base_imagem.isChecked(),
                "sobreposta_transparente": self.check_imagem_sobreposta.isChecked(),
                "minucias_transparentes": self.check_minucias.isChecked(),
                "mostrar_numeros_marcacao": bool(self.mostrar_numeros_marcacao),
                "mostrar_direcao_minucias": bool(self.mostrar_direcao_minucias),
            },
            "deteccao": {
                "quantidade_minucias": getattr(self, "quantidade_minucias", None),
                "distancia_minima": getattr(self, "distancia_minima", None),
            },
            "pontos": {
                "minutiae_points": getattr(self, "minutiae_points", {}),
                "pontos_sobreposta": [],
                "minucias_detectadas_sobreposta": [],
                "pontos_minucia_manual": getattr(self, "pontos_minucia_manual", []),
                "pontos_delta": getattr(self, "pontos_delta", []),
                "pontos_core": getattr(self, "pontos_core", []),
                "correspondencias": getattr(self, "correspondencias", {}),
                "direcoes_minucias_base": getattr(self, "direcoes_minucias_base", {}),
                "direcoes_minucias_sobreposta": getattr(self, "direcoes_minucias_sobreposta", {}),
            },
            "grafos": {
                "grafo_base": getattr(self, "grafo_base", []),
                "grafo_sobreposta": getattr(self, "grafo_sobreposta", []),
                "grafo_matching": getattr(self, "grafo_matching", []),
            },
        }
        return self.normalizar_valor_log(dados)

    def normalizar_valor_log(self, valor):
        """Converte valores comuns do app para tipos aceitos por JSON."""
        if isinstance(valor, dict):
            return {str(k): self.normalizar_valor_log(v) for k, v in valor.items()}
        if isinstance(valor, (list, tuple)):
            return [self.normalizar_valor_log(v) for v in valor]
        if isinstance(valor, set):
            return [self.normalizar_valor_log(v) for v in valor]
        if isinstance(valor, np.integer):
            return int(valor)
        if isinstance(valor, np.floating):
            return float(valor)
        if isinstance(valor, QColor):
            return valor.name()
        if valor is None or isinstance(valor, (str, int, float, bool)):
            return valor
        return str(valor)

    def registrar_log(self, acao, detalhes=None, posicao=None, incluir_estado=True):
        """Registra acoes do usuario e o estado visual para auditoria/reproducao."""
        if not hasattr(self, "log_usuario"):
            self.log_usuario = []

        if isinstance(acao, dict) and detalhes is None:
            detalhes = acao
            acao = detalhes.get("acao", "estado")

        evento = {
            "id": len(self.log_usuario) + 1,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "acao": str(acao),
            "detalhes": self.normalizar_valor_log(detalhes or {}),
        }

        if posicao is not None:
            evento["mouse"] = {
                "x": round(float(posicao.x()), 2),
                "y": round(float(posicao.y()), 2),
            }

        if incluir_estado:
            evento["estado"] = self.normalizar_valor_log(self.obter_estado_log())

        self.log_usuario.append(evento)

    def carregar_dados_log(self, arquivo_log_path):
        """Carrega o snapshot do trabalho a partir de um JSON ou de um ZIP contendo log.json."""
        if arquivo_log_path.lower().endswith(".zip"):
            with zipfile.ZipFile(arquivo_log_path, "r") as zipf:
                nomes = zipf.namelist()
                nome_log = next(
                    (nome for nome in nomes if os.path.basename(nome) in ["log.json", "log_usuario.json"]),
                    None
                )
                if nome_log is None:
                    raise ValueError("O ZIP selecionado nao possui log.json.")

                with zipf.open(nome_log) as f:
                    dados = json.loads(f.read().decode("utf-8"))
        else:
            with open(arquivo_log_path, "r", encoding="utf-8") as f:
                dados = json.load(f)

        if not isinstance(dados, dict):
            raise ValueError("Arquivo de log invalido.")

        return dados

    def extrair_imagem_do_log(self, zip_path, nome_arquivo_zip, temp_dir):
        """Extrai uma imagem do ZIP do log para carregamento temporario."""
        if not zip_path or not nome_arquivo_zip:
            return None

        with zipfile.ZipFile(zip_path, "r") as zipf:
            nome_encontrado = next(
                (nome for nome in zipf.namelist() if os.path.basename(nome) == nome_arquivo_zip),
                None
            )
            if not nome_encontrado:
                return None

            destino = os.path.join(temp_dir, os.path.basename(nome_encontrado))
            with zipf.open(nome_encontrado) as origem, open(destino, "wb") as saida:
                shutil.copyfileobj(origem, saida)
            return destino

    def resolver_imagem_log(self, dados, chave, zip_path, temp_dir):
        """Resolve a imagem salva no ZIP ou, como fallback, no caminho original."""
        info = dados.get("imagens", {}).get(chave, {})

        caminho_zip = self.extrair_imagem_do_log(zip_path, info.get("arquivo_zip"), temp_dir)
        if caminho_zip:
            return caminho_zip

        caminho_original = info.get("arquivo_original")
        if caminho_original and os.path.exists(caminho_original):
            return caminho_original

        return None

    def restaurar_lista_pontos(self, pontos, tamanho=2):
        """Converte listas do JSON em tuplas numericas."""
        restaurados = []
        for ponto in pontos or []:
            if len(ponto) >= tamanho:
                if tamanho == 3:
                    restaurados.append((int(ponto[0]), int(ponto[1]), float(ponto[2])))
                else:
                    restaurados.append((int(ponto[0]), int(ponto[1])))
        return restaurados

    def restaurar_dict_pontos(self, pontos):
        restaurados = {}
        for idx, ponto in (pontos or {}).items():
            if len(ponto) >= 3:
                restaurados[int(idx)] = (int(ponto[0]), int(ponto[1]), float(ponto[2]))
        return restaurados

    def restaurar_dict_tuplas(self, pontos):
        restaurados = {}
        for idx, ponto in (pontos or {}).items():
            if len(ponto) >= 2:
                restaurados[int(idx)] = (float(ponto[0]), float(ponto[1]))
        return restaurados

    def aplicar_dados_log(self, dados, arquivo_log_path):
        """Restaura imagens, pontos detectados, marcações e correspondências do log."""
        if dados.get("tipo") == "benah_movimento_usuario" and "eventos" in dados:
            raise ValueError("Esse log antigo registra movimento do mouse e nao contem os pontos do trabalho.")

        zip_path = arquivo_log_path if arquivo_log_path.lower().endswith(".zip") else None
        temp_dir = tempfile.mkdtemp()

        try:
            caminho_base = self.resolver_imagem_log(dados, "base", zip_path, temp_dir)
            caminho_sobreposta = self.resolver_imagem_log(dados, "sobreposta", zip_path, temp_dir)

            if not caminho_base or not caminho_sobreposta:
                raise ValueError("Nao foi possivel encontrar as imagens base e sobreposta no ZIP ou nos caminhos originais.")

            self.limpar_minucias_base()
            self.limpar_minucias_sobreposta()
            self.correspondencias = {}

            self.carregar_imagem_base_from_path(caminho_base)
            self.carregar_imagem_sobreposta_from_path(caminho_sobreposta)

            visual = dados.get("visual", {})
            self.zoom_factor = float(visual.get("zoom_factor", self.zoom_factor))
            self.base_position_x = float(visual.get("base_position_x", self.base_position_x))
            self.base_position_y = float(visual.get("base_position_y", self.base_position_y))
            self.sobreposta_position_x = float(visual.get("sobreposta_position_x", self.sobreposta_position_x))
            self.sobreposta_position_y = float(visual.get("sobreposta_position_y", self.sobreposta_position_y))
            self.transparencia = float(visual.get("transparencia", self.transparencia))
            self.mostrar_numeros_marcacao = bool(visual.get("mostrar_numeros_marcacao", True))
            self.mostrar_direcao_minucias = bool(visual.get("mostrar_direcao_minucias", False))

            pontos = dados.get("pontos", {})
            self.minutiae_points = self.restaurar_dict_pontos(pontos.get("minutiae_points", {}))
            self.pontos_sobreposta = []
            self.minucias_detectadas_sobreposta = []
            self.pontos_minucia_manual = self.restaurar_lista_pontos(pontos.get("pontos_minucia_manual", []))
            self.pontos_delta = self.restaurar_lista_pontos(pontos.get("pontos_delta", []))
            self.pontos_core = self.restaurar_lista_pontos(pontos.get("pontos_core", []))
            self.correspondencias = self.restaurar_dict_tuplas(pontos.get("correspondencias", {}))
            self.direcoes_minucias_base = self.restaurar_dict_tuplas(pontos.get("direcoes_minucias_base", {}))
            self.direcoes_minucias_sobreposta = self.restaurar_dict_tuplas(pontos.get("direcoes_minucias_sobreposta", {}))
            self.pontos_correspondentes_sobreposta = set(self.correspondencias.values())

            grafos = dados.get("grafos", {})
            self.grafo_base = grafos.get("grafo_base", [])
            self.grafo_sobreposta = grafos.get("grafo_sobreposta", [])
            self.grafo_matching = grafos.get("grafo_matching", [])

            if self.base_pixmap:
                self.base_original = self.base_pixmap.copy()

            self.minucias_detectadas = bool(self.minutiae_points)
            if self.minutiae_points:
                self.check_minucias.setEnabled(True)
                self.redesenhar_minucias()

            self.check_base_imagem.setChecked(bool(visual.get("base_transparente", False)))
            self.check_imagem_sobreposta.setChecked(bool(visual.get("sobreposta_transparente", False)))
            self.check_minucias.setChecked(bool(visual.get("minucias_transparentes", False)))
            self.atualizar_imagens_por_zoom_log()
            self.atualizar_status_lateral()
            self.update()
        finally:
            shutil.rmtree(temp_dir)

    def atualizar_imagens_por_zoom_log(self):
        """Recalcula pixmaps escalados usando o zoom restaurado do log."""
        if self.base_pixmap:
            pixmap_base = self.base_colored if self.base_colored else self.base_pixmap
            self.base_scaled = pixmap_base.scaled(
                max(1, int(self.base_pixmap.width() * self.zoom_factor)),
                max(1, int(self.base_pixmap.height() * self.zoom_factor)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if self.sobreposta_pixmap:
            pixmap_sobreposta = self.sobreposta_colored if self.sobreposta_colored else self.sobreposta_pixmap
            self.sobreposta_scaled = pixmap_sobreposta.scaled(
                max(1, int(self.sobreposta_pixmap.width() * self.zoom_factor)),
                max(1, int(self.sobreposta_pixmap.height() * self.zoom_factor)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if hasattr(self, 'minucias_pixmap') and self.minucias_pixmap:
            self.minucias_scaled = self.minucias_pixmap.scaled(
                max(1, int(self.minucias_pixmap.width() * self.zoom_factor)),
                max(1, int(self.minucias_pixmap.height() * self.zoom_factor)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

    def salvar_imagem_e_dados(self):
        """Salva imagens, minúcias, grafos e matching em um .zip, somente se todas as condições forem atendidas."""
        if self.base_pixmap is None:
            QMessageBox.warning(self, "Erro", "Carregue a imagem base antes de salvar.")
            return

        if self.sobreposta_pixmap is None:
            QMessageBox.warning(self, "Erro", "Carregue a imagem sobreposta antes de salvar.")
            return

        if not self.minucias_detectadas:
            QMessageBox.warning(self, "Erro", "Detecte os pontos antes de salvar.")
            return

        if not self.pontos_delta:
            QMessageBox.warning(self, "Erro", "Marque pelo menos 1 ponto delta antes de salvar.")
            return
        
        if not self.pontos_core:
            QMessageBox.warning(self, "Erro", "Marque pelo menos 1 ponto core antes de salvar.")
            return

        core_ok = any((2000 + i + 1) in getattr(self, 'correspondencias', {}) for i in range(len(self.pontos_core)))
        if not core_ok:
            QMessageBox.warning(self, "Erro", "Você precisa marcar a correspondência do ponto core.")
            return

        delta_ok = any((1000 + i + 1) in getattr(self, 'correspondencias', {}) for i in range(len(self.pontos_delta)))
        if not delta_ok:
            QMessageBox.warning(self, "Erro", "Você precisa marcar a correspondência do ponto delta.")
            return

        total_minucias = list(getattr(self, 'minutiae_points', {}).keys()) + [
            len(getattr(self, 'minutiae_points', {})) + i + 1 for i in range(len(self.pontos_minucia_manual))
        ]
        minucia_ok = any(idx in getattr(self, 'correspondencias', {}) for idx in total_minucias)
        if not minucia_ok:
            QMessageBox.warning(self, "Erro", "Você precisa marcar a correspondência de pelo menos uma minúcia.")
            return

        zip_path = escolher_arquivo_para_salvar(self, "Salvar como ZIP", "Arquivo ZIP (*.zip)")
        if not zip_path:
            return
        if not zip_path.endswith(".zip"):
            zip_path += ".zip"

        self.registrar_log("salvar_zip_iniciado", {"arquivo": zip_path})

        temp_dir = tempfile.mkdtemp()
        try:
            self.base_pixmap.save(os.path.join(temp_dir, "imagem_base_original.png"), "PNG")
            self.sobreposta_pixmap.save(os.path.join(temp_dir, "imagem_sobreposta_original.png"), "PNG")

            log_path = os.path.join(temp_dir, "log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.montar_dados_log(), f, indent=2, ensure_ascii=False)

            def coletar_pontos_base():
                pontos = []
                if hasattr(self, 'minutiae_points'):
                    for idx, (x, y, score) in self.minutiae_points.items():
                        pontos.append((int(x), int(y), "minucia_auto", score, idx))
                
                if hasattr(self, 'pontos_minucia_manual'):
                    for i, (x, y) in enumerate(self.pontos_minucia_manual):
                        idx = len(getattr(self, 'minutiae_points', {})) + i + 1
                        pontos.append((int(x), int(y), "minucia_manual", 1.0, idx))
                
                if hasattr(self, 'pontos_delta'):
                    for i, (x, y) in enumerate(self.pontos_delta):
                        idx = 1000 + i + 1
                        pontos.append((int(x), int(y), "delta", 1.0, idx))
                
                if hasattr(self, 'pontos_core'):
                    for i, (x, y) in enumerate(self.pontos_core):
                        idx = 2000 + i + 1
                        pontos.append((int(x), int(y), "core", 1.0, idx))
                
                return pontos

            def coletar_pontos_sobreposta():
                pontos = []
                if hasattr(self, 'correspondencias'):
                    for idx, (x, y) in self.correspondencias.items():
                        if idx < 1000:
                            tipo = "minucia"
                        elif idx < 2000:
                            tipo = "delta"
                        else:
                            tipo = "core"
                        pontos.append((int(x), int(y), tipo, 1.0, idx))
                return pontos

            pontos_base = coletar_pontos_base()
            pontos_sobreposta = coletar_pontos_sobreposta()

            min_base_path = os.path.join(temp_dir, "minucias_base.txt")
            with open(min_base_path, "w") as f:
                f.write("# Mapa de Minúcias da Base\n")
                f.write("# Formato: x, y, tipo, id\n")
                for x, y, tipo, score, idx in pontos_base:
                    f.write(f"{x}, {y}, {tipo}, {idx}\n")

            min_sobreposta_path = os.path.join(temp_dir, "minucias_sobreposta.txt")
            with open(min_sobreposta_path, "w") as f:
                f.write("# Mapa de Minúcias da Sobreposta\n")
                f.write("# Formato: x, y, tipo, id_correspondente\n")
                for x, y, tipo, score, idx in pontos_sobreposta:
                    f.write(f"{x}, {y}, {tipo}, {idx}\n")

            dir_base_path = os.path.join(temp_dir, "direcoes_base.txt")
            with open(dir_base_path, "w") as f:
                f.write("# Direções das Minúcias da Base\n")
                f.write("# Formato: id, dx, dy\n")
                if hasattr(self, 'direcoes_minucias_base'):
                    for idx, (dx, dy) in self.direcoes_minucias_base.items():
                        f.write(f"{idx}, {dx:.4f}, {dy:.4f}\n")

            dir_sobreposta_path = os.path.join(temp_dir, "direcoes_sobreposta.txt")
            with open(dir_sobreposta_path, "w") as f:
                f.write("# Direções das Minúcias da Sobreposta\n")
                f.write("# Formato: id, dx, dy\n")
                if hasattr(self, 'direcoes_minucias_sobreposta'):
                    for idx, (dx, dy) in self.direcoes_minucias_sobreposta.items():
                        f.write(f"{idx}, {dx:.4f}, {dy:.4f}\n")

            img_min_base = QPixmap(self.base_pixmap.size())
            img_min_base.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img_min_base)
            
            for x, y, tipo, score, idx in pontos_base:
                if tipo in ["minucia_auto", "minucia_manual"]:
                    r, g, b = self.cor_minucias
                    cor = QColor(r, g, b)
                elif tipo == "delta":
                    cor = self.cor_delta
                elif tipo == "core":
                    cor = self.cor_core
                
                painter.setBrush(cor)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x - self.raio_minucias), int(y - self.raio_minucias), 
                                int(self.raio_minucias * 2), int(self.raio_minucias * 2))
            
            painter.end()
            img_min_base.save(os.path.join(temp_dir, "imagem_minucias_base.png"), "PNG")

            img_min_sobreposta = QPixmap(self.sobreposta_pixmap.size())
            img_min_sobreposta.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img_min_sobreposta)
            
            for x, y, tipo, score, idx in pontos_sobreposta:
                if tipo == "minucia":
                    r, g, b = self.cor_minucias
                    cor = QColor(r, g, b)
                elif tipo == "delta":
                    cor = self.cor_delta
                elif tipo == "core":
                    cor = self.cor_core
                
                painter.setBrush(cor)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x - self.raio_minucias), int(y - self.raio_minucias), 
                                int(self.raio_minucias * 2), int(self.raio_minucias * 2))
            
            painter.end()
            img_min_sobreposta.save(os.path.join(temp_dir, "imagem_minucias_sobreposta.png"), "PNG")

            if hasattr(self, 'mostrar_direcao_minucias') and self.mostrar_direcao_minucias and hasattr(self, 'direcoes_minucias_base'):
                img_dir_base = QPixmap(self.base_pixmap.size())
                img_dir_base.fill(Qt.GlobalColor.transparent)
                painter = QPainter(img_dir_base)
                pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                painter.setPen(pen)
                
                for x, y, tipo, score, idx in pontos_base:
                    if tipo in ["minucia_auto", "minucia_manual"] and idx in self.direcoes_minucias_base:
                        dx, dy = self.direcoes_minucias_base[idx]
                        end_x = int(x + dx * self.tamanho_direcao_altura)
                        end_y = int(y + dy * self.tamanho_direcao_altura)
                        painter.drawLine(int(x), int(y), end_x, end_y)
                
                painter.end()
                img_dir_base.save(os.path.join(temp_dir, "imagem_direcoes_base.png"), "PNG")

            if hasattr(self, 'mostrar_direcao_minucias') and self.mostrar_direcao_minucias and hasattr(self, 'direcoes_minucias_sobreposta'):
                img_dir_sobreposta = QPixmap(self.sobreposta_pixmap.size())
                img_dir_sobreposta.fill(Qt.GlobalColor.transparent)
                painter = QPainter(img_dir_sobreposta)
                pen = QPen(self.cor_direcao_minucias, self.tamanho_direcao_espessura)
                painter.setPen(pen)
                
                for x, y, tipo, score, idx in pontos_sobreposta:
                    if hasattr(self, 'direcoes_minucias_sobreposta') and idx in self.direcoes_minucias_sobreposta:
                        dx, dy = self.direcoes_minucias_sobreposta[idx]
                        end_x = int(x + dx * self.tamanho_direcao_altura)
                        end_y = int(y + dy * self.tamanho_direcao_altura)
                        painter.drawLine(int(x), int(y), end_x, end_y)
                
                painter.end()
                img_dir_sobreposta.save(os.path.join(temp_dir, "imagem_direcoes_sobreposta.png"), "PNG")

            def salvar_grafos_csv(pontos, nome_arquivo, temp_dir):
                if len(pontos) < 2:
                    return
                
                csv_path = os.path.join(temp_dir, nome_arquivo)
                with open(csv_path, "w", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(["tipo_grafo", "x1", "y1", "x2", "y2"])
                    
                    # MST
                    G = nx.Graph()
                    for idx, (x, y, tipo_ponto, score, id_ponto) in enumerate(pontos):
                        G.add_node(idx, pos=(x, y))
                    for i in range(len(pontos)):
                        for j in range(i + 1, len(pontos)):
                            x1, y1 = pontos[i][0], pontos[i][1]
                            x2, y2 = pontos[j][0], pontos[j][1]
                            peso = math.hypot(x2 - x1, y2 - y1)
                            G.add_edge(i, j, weight=peso)
                    mst = nx.minimum_spanning_tree(G)
                    for u, v in mst.edges:
                        x1, y1 = pontos[u][0], pontos[u][1]
                        x2, y2 = pontos[v][0], pontos[v][1]
                        writer.writerow(["MST", x1, y1, x2, y2])
                    
                    # E-ball (substituindo GC)
                    # Calcula o raio para conectar os pontos mais próximos
                    distancias = []
                    for i in range(len(pontos)):
                        for j in range(i + 1, len(pontos)):
                            x1, y1 = pontos[i][0], pontos[i][1]
                            x2, y2 = pontos[j][0], pontos[j][1]
                            dist = math.hypot(x2 - x1, y2 - y1)
                            distancias.append(dist)
                    
                    if distancias:
                        # Usa uma fração das distâncias médias como raio
                        raio_eball = sum(distancias) / len(distancias) * 0.7  # 70% da distância média
                        
                        for i in range(len(pontos)):
                            for j in range(i + 1, len(pontos)):
                                x1, y1 = pontos[i][0], pontos[i][1]
                                x2, y2 = pontos[j][0], pontos[j][1]
                                dist = math.hypot(x2 - x1, y2 - y1)
                                if dist <= raio_eball:
                                    writer.writerow(["E-ball", x1, y1, x2, y2])
                    
                    # K-NN
                    k = min(3, len(pontos) - 1)
                    for i, (x1, y1, _, _, _) in enumerate(pontos):
                        distancias = [(j, math.hypot(x1 - x2, y1 - y2)) for j, (x2, y2, _, _, _) in enumerate(pontos) if j != i]
                        distancias.sort(key=lambda x: x[1])
                        for j, _ in distancias[:k]:
                            x2, y2 = pontos[j][0], pontos[j][1]
                            writer.writerow(["KNN", x1, y1, x2, y2])
                    
                    # RWC
                    indices = list(range(len(pontos)))
                    random.shuffle(indices)
                    for i in range(len(indices) - 1):
                        u = indices[i]
                        v = indices[i + 1]
                        x1, y1 = pontos[u][0], pontos[u][1]
                        x2, y2 = pontos[v][0], pontos[v][1]
                        writer.writerow(["RWC", x1, y1, x2, y2])

            # Gerar os CSVs dos grafos
            if len(pontos_base) >= 2:
                salvar_grafos_csv(pontos_base, "grafos_base.csv", temp_dir)

            if len(pontos_sobreposta) >= 2:
                salvar_grafos_csv(pontos_sobreposta, "grafos_sobreposta.csv", temp_dir)

            if hasattr(self, 'correspondencias') and self.correspondencias:
                largura_total = self.base_pixmap.width() + self.sobreposta_pixmap.width()
                altura_total = max(self.base_pixmap.height(), self.sobreposta_pixmap.height())
                
                img_matching = QPixmap(largura_total, altura_total)
                img_matching.fill(Qt.GlobalColor.transparent)
                painter = QPainter(img_matching)
                
                for x, y, tipo, score, idx in pontos_base:
                    if tipo in ["minucia_auto", "minucia_manual"]:
                        r, g, b = self.cor_minucias
                        cor = QColor(r, g, b)
                    elif tipo == "delta":
                        cor = self.cor_delta
                    elif tipo == "core":
                        cor = self.cor_core
                    
                    painter.setBrush(cor)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(int(x - self.raio_minucias), int(y - self.raio_minucias), 
                                    int(self.raio_minucias * 2), int(self.raio_minucias * 2))
                
                for x, y, tipo, score, idx in pontos_sobreposta:
                    if tipo == "minucia":
                        r, g, b = self.cor_minucias
                        cor = QColor(r, g, b)
                    elif tipo == "delta":
                        cor = self.cor_delta
                    elif tipo == "core":
                        cor = self.cor_core
                    
                    painter.setBrush(cor)
                    painter.setPen(Qt.PenStyle.NoPen)
                    x_offset = int(x + self.base_pixmap.width())
                    painter.drawEllipse(int(x_offset - self.raio_minucias), int(y - self.raio_minucias), 
                                    int(self.raio_minucias * 2), int(self.raio_minucias * 2))
                
                pen = QPen(self.cor_matching, self.espessura_matching)
                painter.setPen(pen)
                
                for idx, (x2, y2) in self.correspondencias.items():
                    x1, y1 = None, None
                    
                    if idx < 1000:  
                        if hasattr(self, 'minutiae_points') and idx in self.minutiae_points:
                            x1, y1, _ = self.minutiae_points[idx]
                        else:
                            offset = len(getattr(self, 'minutiae_points', {}))
                            i = idx - offset - 1
                            if hasattr(self, 'pontos_minucia_manual') and 0 <= i < len(self.pontos_minucia_manual):
                                x1, y1 = self.pontos_minucia_manual[i]
                    elif 1000 <= idx < 2000:  
                        i = idx - 1000 - 1
                        if hasattr(self, 'pontos_delta') and 0 <= i < len(self.pontos_delta):
                            x1, y1 = self.pontos_delta[i]
                    elif idx >= 2000:  
                        i = idx - 2000 - 1
                        if hasattr(self, 'pontos_core') and 0 <= i < len(self.pontos_core):
                            x1, y1 = self.pontos_core[i]
                    
                    if x1 is not None and y1 is not None:
                        x2_offset = int(x2 + self.base_pixmap.width())
                        painter.drawLine(int(x1), int(y1), x2_offset, int(y2))
                
                painter.end()
                img_matching.save(os.path.join(temp_dir, "matching_conexoes.png"), "PNG")

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename in os.listdir(temp_dir):
                    zipf.write(os.path.join(temp_dir, filename), arcname=filename)

            self.registrar_log("salvar_zip_concluido", {"arquivo": zip_path})
            QMessageBox.information(self, "Sucesso", "Todos os dados foram salvos com sucesso em um arquivo .zip!")

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar: {e}")
        finally:
            shutil.rmtree(temp_dir)

    def atualizar_transparencia(self):
        """Atualiza a transparência das imagens com base no valor do slider."""
        valor_transparencia = self.slider3.value() / 100.0
        self.transparencia = valor_transparencia
        self.update()  
    
    def mudar_cor_do_retangulo(self):
        """Abre um diálogo para alterar a cor de fundo do layout."""
        if not hasattr(self, 'cor_dialog') or not self.cor_dialog.isVisible():
            self.cor_dialog = CorDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.cor_dialog)
        else:
            self.cor_dialog.raise_()
            self.cor_dialog.activateWindow()

    def carregar(self):
        """Abre um diálogo para carregar novas imagens base e sobreposta."""
        if not hasattr(self, 'carregar_dialog') or not self.carregar_dialog.isVisible():
            self.carregar_dialog = Carregar(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.carregar_dialog)
        else:
            self.carregar_dialog.raise_()
            self.carregar_dialog.activateWindow()

    def activate_easter_egg(self):
        """Ativa um Easter Egg, substituindo a imagem base por uma imagem especial."""
        self.base_scaled = None
        self.sobreposta_scaled = None
        self.base_position_x = int(119 * self.scale_x)
        self.base_position_y = int(30 * self.scale_y)

        if not self.easter_egg_image.isNull():
            self.easter_egg_image = self.easter_egg_image.scaled(
                int(self.easter_egg_image.width() * (0.9 * self.scale_x) ),
                int(self.easter_egg_image.height() * (0.9 * self.scale_y)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.easter_egg_active = True
            self.update()  
            self.sound_effect.play()
        else:
            print("Erro: Imagem do easter egg não foi carregada.")

    def deactivate_easter_egg(self):
        """Desativa o Easter Egg, retornando à exibição normal."""
        self.easter_egg_active = False 
        self.update()  
        self.sound_effect.stop()

    def open_manual(self):
        """Abre a janela do manual do usuário."""
        if not hasattr(self, 'manual_dialog') or not self.manual_dialog.isVisible():
            self.manual_dialog = ManualDialog(self, self.scale_x, self.scale_y)  
            self.abrir_janela(self.manual_dialog)
        else:
            self.manual_dialog.raise_()
            self.manual_dialog.activateWindow()

    def minimize_app(self):
        """Minimiza a janela principal da aplicação."""
        self.showMinimized()

    def agradecimentos(self):
        """Abre a janela de agradecimentos aos desenvolvedores e colaboradores."""
        if not hasattr(self, 'agradecimentos_dialog') or not self.agradecimentos_dialog.isVisible():
            self.agradecimentos_dialog = AgradecimentosDialog(self, self.scale_x, self.scale_y)  
            self.abrir_janela(self.agradecimentos_dialog)
        else:
            self.agradecimentos_dialog.raise_()
            self.agradecimentos_dialog.activateWindow()
        
    def close_app(self):
        """Fecha a aplicação."""
        self.fechar_janela(self)

    def carregar_cor_fundo(self):
        """Carrega a cor do fundo salva e aplica na interface."""
        saved_color = self.settings.value("cor_fundo", QColor(71, 142, 213))  
        self.rect_color = QColor(saved_color)
        self.update()

    def excluir_imagem_base(self):
        """Remove completamente a imagem base e reseta o estado para inicial."""
        if self.base_pixmap is None:
            QMessageBox.warning(self, "Erro", "Nenhuma imagem base carregada para excluir.")
            return

        self.limpar_minucias_base()
        self.base_pixmap = None
        self.base_scaled = None
        self.base_colored = None
        self.cor_base_salva = None
        self.grafo_base = []

        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.settings.setValue("checkbox_base", False)  
        self.settings.sync()

        self.history.clear()
        self.current_index = -1

        self.check_base_imagem.setChecked(False)
        self.check_base_imagem.setEnabled(False)

        self.update()
        self.repaint()

        self.sincronizar_imagens()
        self.imagem_base_ok.emit(False)
        self.atualizar_status_lateral()
        self.registrar_log("excluir_imagem_base")

    def excluir_imagem_sobreposta(self):
        """Remove completamente a imagem sobreposta e reseta o estado para inicial."""
        if self.sobreposta_pixmap is None:
            QMessageBox.warning(self, "Erro", "Nenhuma imagem sobreposta carregada para excluir.")
            return

        self.limpar_minucias_sobreposta()
        self.sobreposta_pixmap = None
        self.sobreposta_scaled = None
        self.sobreposta_colored = None
        self.sobreposta_rotated = None  
        self.cor_sobreposta_salva = None
        self.cor_sobreposta_alterada = False
        self.grafo_sobreposta = []

        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        self.settings.sync()

        self.history.clear()
        self.current_index = -1

        self.update()
        self.repaint()
        self.sincronizar_imagens()
        self.imagem_sobreposta_carregada.emit(False)  
        self.settings.setValue("checkbox_sobreposta", False) 
        self.settings.sync()
        self.atualizar_status_lateral()
        self.registrar_log("excluir_imagem_sobreposta")
    
    def verificar_sobreposta_carregada(self):
        """Verifica se a imagem sobreposta está carregada antes de permitir alterações."""
        if not hasattr(self.parent(), 'sobreposta_scaled') or self.parent().sobreposta_scaled is None:
            QMessageBox.warning(self, "Erro", "A imagem sobreposta não foi carregada!")
            return False
        return True
    
    def abrir_janela(self, janela):
        """Abre uma nova janela e controla o número de janelas abertas."""
        self.janelas_abertas = [j for j in self.janelas_abertas if not j.isHidden()]

        if janela in self.janelas_abertas:
            return 

        if len(self.janelas_abertas) >= self.max_janelas:
            primeira_janela = self.janelas_abertas.pop(0)
            primeira_janela.close()

        self.janelas_abertas.append(janela)
        posicionar_dialogo_no_parent(janela, self)
        janela.show()

    def fechar_janela(self, janela):
        """Fecha uma janela e remove do controle."""
        if janela in self.janelas_abertas:
            self.janelas_abertas.remove(janela)
        janela.close()
    
    def limpar_cache(self):
        """Remove todas as configurações salvas, garantindo um reset após um crash."""
        self.settings.remove("cor_base")
        self.settings.remove("cor_sobreposta")
        self.settings.setValue("checkbox_base", False)
        self.settings.setValue("checkbox_sobreposta", False)
        self.settings.sync()

        self.cor_base_salva = None
        self.cor_sobreposta_salva = None
    
    def abrir_janela_log(self):
        """Abre a janela de log para restaurar um trabalho salvo."""
        if not hasattr(self, 'janela_log') or not self.janela_log.isVisible():
            self.janela_log = JanelaLogDialog(self, self.scale_x, self.scale_y)
            self.janela_log.confirmar_aplicacao.connect(self.aplicar_log_automaticamente)
            self.abrir_janela(self.janela_log)
        else:
            self.janela_log.raise_()
            self.janela_log.activateWindow()

    def aplicar_log_automaticamente(self, arquivo_log_path):
        """Restaura imagens, pontos e correspondencias registrados no ZIP salvo ou em um JSON."""
        try:
            dados = self.carregar_dados_log(arquivo_log_path)
            self.aplicar_dados_log(dados, arquivo_log_path)
            self.registrar_log("carregar_log_zip", {"arquivo": arquivo_log_path})
            QMessageBox.information(self, "Log", "Log carregado com imagens, pontos e correspondencias.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar log: {e}")
        
    def mudar_cor_sobreposta_automaticamente(self):
        """Muda automaticamente a cor da imagem sobreposta para uma cor aleatória sem abrir a janela de diálogo."""
        if not hasattr(self, 'cor_dialog') or self.cor_dialog is None:
            self.cor_dialog = CorDialog(self)

        if self.sobreposta_scaled:
            while True:
                cor = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if cor != Qt.GlobalColor.white and cor != Qt.GlobalColor.black:
                    break

            colored_pixmap = self.cor_dialog.aplicar_cor_na_imagem(self.sobreposta_scaled, cor)

            self.sobreposta_colored = colored_pixmap.copy()
            self.sobreposta_scaled = self.sobreposta_colored.copy()

            self.salvar_cor_aplicada(cor, "sobreposta")

            self.cor_sobreposta_alterada = True
            self.update()

            return cor  
        
    def mudar_cor_base_automaticamente(self, cor_sobreposta):
        """Muda automaticamente a cor da imagem base para uma cor aleatória diferente da sobreposta."""
        if not hasattr(self, 'cor_dialog') or self.cor_dialog is None:
            self.cor_dialog = CorDialog(self)

        if self.base_scaled:
            while True:
                cor = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if cor != cor_sobreposta:
                    break

            colored_pixmap = self.cor_dialog.aplicar_cor_na_imagem(self.base_scaled, cor)

            self.base_colored = colored_pixmap.copy()
            self.base_scaled = self.base_colored.copy()

            self.salvar_cor_aplicada(cor, "base")
            self.cor_base_salva = cor
            self.update()

    def carregar_imagem_base_from_path(self, arquivo_imagem):
        """Carrega a imagem base diretamente a partir do caminho fornecido."""
        if not os.path.exists(arquivo_imagem):
            QMessageBox.warning(self, "Erro", f"Imagem base não encontrada: {arquivo_imagem}")
            return

        self.arquivo_base = arquivo_imagem
        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        imagem_cv = cv2.imread(arquivo_imagem, cv2.IMREAD_UNCHANGED)

        if imagem_cv is None:
            QMessageBox.warning(self, "Erro", "Erro ao carregar a imagem base.")
            return

        if imagem_cv.shape[2] == 4:
            altura, largura, _ = imagem_cv.shape
            qimage = QImage(imagem_cv.data, largura, altura, 4 * largura, QImage.Format.Format_RGBA8888)
        else:
            imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
            altura, largura, _ = imagem_rgb.shape
            qimage = QImage(imagem_rgb.data, largura, altura, 3 * largura, QImage.Format.Format_RGB888)

        self.base_pixmap = QPixmap.fromImage(qimage)

        if self.cor_base_salva:
            cor_dialog = CorDialog(self)
            self.base_colored = cor_dialog.aplicar_cor_na_imagem(self.base_pixmap.copy(), self.cor_base_salva)
        else:
            self.base_colored = None

        self.base_scaled = self.base_pixmap.scaled(
            int(self.base_pixmap.width() * self.zoom_factor),
            int(self.base_pixmap.height() * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.check_base_imagem.setEnabled(True)
        self.alterar_transparencia_imagem_base()
        self.imagem_base_ok.emit(True)
        self.settings.setValue("checkbox_base", True)
        self.settings.sync()
        self.atualizar_status_lateral()
        self.update()

    def carregar_imagem_base_automatica(self, caminho):
        """Carrega uma imagem base automaticamente a partir de um caminho especificado."""
        if not os.path.exists(caminho):
            QMessageBox.warning(self, "Erro", f"Imagem base não encontrada: {caminho}")
            return

        self.arquivo_base = caminho
        self.carregar_imagem_base_from_path(caminho)

    def carregar_imagem_sobreposta_from_path(self, arquivo_imagem):
        """Carrega a imagem sobreposta diretamente a partir do caminho fornecido."""
        if not os.path.exists(arquivo_imagem):
            QMessageBox.warning(self, "Erro", f"Imagem sobreposta não encontrada: {arquivo_imagem}")
            return

        self.arquivo_sobreposta = arquivo_imagem
        self.zoom_factor = 0.40 * min(self.scale_x, self.scale_y)

        imagem_cv = cv2.imread(arquivo_imagem, cv2.IMREAD_UNCHANGED)

        if imagem_cv is None:
            QMessageBox.warning(self, "Erro", "Erro ao carregar a imagem sobreposta.")
            return

        if imagem_cv.shape[2] == 4:
            altura, largura, _ = imagem_cv.shape
            qimage = QImage(imagem_cv.data, largura, altura, 4 * largura, QImage.Format.Format_RGBA8888)
        else:
            imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
            altura, largura, _ = imagem_rgb.shape
            qimage = QImage(imagem_rgb.data, largura, altura, 3 * largura, QImage.Format.Format_RGB888)

        self.sobreposta_pixmap = QPixmap.fromImage(qimage)

        if self.cor_sobreposta_salva:
            cor_dialog = CorDialog(self)
            self.sobreposta_colored = cor_dialog.aplicar_cor_na_imagem(self.sobreposta_pixmap.copy(), self.cor_sobreposta_salva)
        else:
            self.sobreposta_colored = None

        self.sobreposta_scaled = self.sobreposta_colored.copy() if self.sobreposta_colored else self.sobreposta_pixmap.scaled(
            int(self.sobreposta_pixmap.width() * self.zoom_factor),
            int(self.sobreposta_pixmap.height() * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.check_imagem_sobreposta.setEnabled(True)
        self.alterar_transparencia_imagem_sobreposta()
        self.imagem_sobreposta_carregada.emit(True)
        self.settings.setValue("checkbox_sobreposta", True)
        self.settings.sync()
        self.atualizar_status_lateral()
        self.update()

    def carregar_imagem_sobreposta_automatica(self, caminho):
        """Carrega uma imagem sobreposta automaticamente a partir de um caminho especificado."""
        if not os.path.exists(caminho):
            QMessageBox.warning(self, "Erro", f"Imagem sobreposta não encontrada: {caminho}")
            return

        self.arquivo_sobreposta = caminho
        self.carregar_imagem_sobreposta_from_path(caminho)

    def open_transparency_dialog(self):
        """Abre a janela de controle de transparência."""
        if not hasattr(self, 'transparencia_dialog') or not self.transparencia_dialog.isVisible():
            self.transparencia_dialog = TransparenciaDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.transparencia_dialog)
        else:
            self.transparencia_dialog.raise_()
            self.transparencia_dialog.activateWindow()

    def abrir_janela_detectar_minucias(self):
        if not hasattr(self, 'marcar_dialog') or not self.marcar_dialog.isVisible():
            self.marcar_dialog = MarcarDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.marcar_dialog)
        else:
            self.marcar_dialog.raise_()
            self.marcar_dialog.activateWindow()
    
    def limpar_minucias_base(self):
        """Remove todas as minúcias e marcações relacionadas à imagem base."""
        self.minucias_pixmap = None
        self.minucias_scaled = None
        self.minutiae_points = {}
        self.pontos_minucia_manual = []
        self.pontos_delta = []
        self.pontos_core = []
        self.grafo_base = []
        self.minucia_selecionada = None
        self.minucias_detectadas = False
        self.check_minucias.setChecked(False)
        self.check_minucias.setEnabled(False)
        self.atualizar_status_lateral()
        self.update()

    def limpar_minucias_sobreposta(self):
        """Remove todas as minúcias e correspondências da imagem sobreposta."""
        self.pontos_sobreposta = []
        self.minucias_detectadas_sobreposta = []
        self.pontos_delta_sobreposta_detectados = []
        self.pontos_core_sobreposta_detectados = []
        self.pontos_correspondentes_sobreposta = set()
        if hasattr(self, "correspondencias"):
            self.correspondencias = {k: v for k, v in self.correspondencias.items() if k < 1000}  
        self.grafo_sobreposta = []
        self.atualizar_status_lateral()
        self.update()

    def gerar_grafo_minucias(self, tipo=0):
        """Gera o grafo (de tipo selecionado) para a imagem base e sobreposta."""

        def coletar_todos_os_pontos():
            pontos = []

            if hasattr(self, 'minutiae_points'):
                pontos += [tuple(p[:2]) for p in self.minutiae_points.values()]

            if hasattr(self, 'pontos_minucia_manual'):
                pontos += self.pontos_minucia_manual

            if hasattr(self, 'pontos_delta'):
                pontos += self.pontos_delta

            if hasattr(self, 'pontos_core'):
                pontos += self.pontos_core

            return pontos

        def coletar_todos_os_pontos_sobreposta():
            pontos = []
            
            # CORREÇÃO: Adiciona TODAS as minúcias detectadas automaticamente
            if hasattr(self, 'pontos_sobreposta'):
                pontos += [(x, y) for x, y, _ in self.pontos_sobreposta]
            
            # Adiciona as correspondências marcadas manualmente (que não estão nas automáticas)
            if hasattr(self, 'correspondencias'):
                for idx, (x, y) in self.correspondencias.items():
                    ponto = (x, y)
                    # Evita duplicatas verificando se o ponto já existe
                    if not any(abs(ponto[0] - p[0]) < 1 and abs(ponto[1] - p[1]) < 1 for p in pontos):
                        pontos.append(ponto)
            
            return pontos

        def gerar_tipo_de_grafo(pontos, tipo=0):
            G = nx.Graph()
            for idx, (x, y) in enumerate(pontos):
                G.add_node(idx, pos=(x, y))

            if tipo == 0:  # MST
                for i in range(len(pontos)):
                    for j in range(i + 1, len(pontos)):
                        x1, y1 = pontos[i]
                        x2, y2 = pontos[j]
                        peso = math.hypot(x2 - x1, y2 - y1)
                        G.add_edge(i, j, weight=peso)
                mst = nx.minimum_spanning_tree(G)
                G = mst  

            elif tipo == 1:  # E-ball (substituindo GC)
                # Calcula todas as distâncias para determinar o raio
                distancias = []
                for i in range(len(pontos)):
                    for j in range(i + 1, len(pontos)):
                        x1, y1 = pontos[i]
                        x2, y2 = pontos[j]
                        dist = math.hypot(x2 - x1, y2 - y1)
                        distancias.append(dist)
                
                if distancias:
                    # Define o raio como 70% da distância média
                    raio_eball = sum(distancias) / len(distancias) * 0.7
                    
                    # Conecta pontos dentro do raio
                    for i in range(len(pontos)):
                        for j in range(i + 1, len(pontos)):
                            x1, y1 = pontos[i]
                            x2, y2 = pontos[j]
                            dist = math.hypot(x2 - x1, y2 - y1)
                            if dist <= raio_eball:
                                G.add_edge(i, j)

            elif tipo == 2:  # K-NN
                k = min(3, len(pontos) - 1)
                for i, (x1, y1) in enumerate(pontos):
                    distancias = [(j, math.hypot(x1 - x2, y1 - y2)) for j, (x2, y2) in enumerate(pontos) if j != i]
                    distancias.sort(key=lambda x: x[1])
                    for j, _ in distancias[:k]:
                        G.add_edge(i, j)

            elif tipo == 3:  # RWC
                indices = list(range(len(pontos)))
                random.shuffle(indices)
                for i in range(len(indices) - 1):
                    u = indices[i]
                    v = indices[i + 1]
                    G.add_edge(u, v)

            componentes = list(nx.connected_components(G))
            if len(componentes) > 1:
                for i in range(len(componentes) - 1):
                    n1 = list(componentes[i])[0]
                    n2 = list(componentes[i + 1])[0]
                    G.add_edge(n1, n2)

            return [((G.nodes[u]["pos"]), (G.nodes[v]["pos"])) for u, v in G.edges]

        pontos_base = coletar_todos_os_pontos()
        pontos_sobreposta = coletar_todos_os_pontos_sobreposta()

        self.grafo_base = gerar_tipo_de_grafo(pontos_base, tipo)
        self.grafo_sobreposta = gerar_tipo_de_grafo(pontos_sobreposta, tipo)

        if hasattr(self, 'grafo_dialog'):
            self.grafo_dialog.ativar_checkbox_grafo()

        self.atualizar_status_lateral()
        self.update()
        
    def abrir_janela_grafo(self):
        """Abre a janela de configurações do grafo."""
        if not hasattr(self, 'grafo_dialog') or not self.grafo_dialog.isVisible():
            self.grafo_dialog = GrafoDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.grafo_dialog)
        else:
            self.grafo_dialog.raise_()
            self.grafo_dialog.activateWindow()

    def abrir_janela_atributos(self):
        """Abre a janela com as opções de atributos das minúcias."""
        if not hasattr(self, 'atributos_dialog') or not self.atributos_dialog.isVisible():
            self.atributos_dialog = AtributosDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.atributos_dialog)
        else:
            self.atributos_dialog.raise_()
            self.atributos_dialog.activateWindow()

    def abrir_janela_matching(self):
        """Abre a janela de opções de Matching."""
        if not hasattr(self, 'matching_dialog') or not self.matching_dialog.isVisible():
            self.matching_dialog = MatchingDialog(self, self.scale_x, self.scale_y)
            self.abrir_janela(self.matching_dialog)
        else:
            self.matching_dialog.raise_()
            self.matching_dialog.activateWindow()

class AgradecimentosDialog(QDialog):
    """Classe que exibe uma janela de agradecimentos, reconhecendo contribuições e suporte fornecido por indivíduos e instituições."""
    def __init__(self, parent, scale_x, scale_y):
        """Inicializa a janela de agradecimentos, configurando a interface gráfica e exibindo mensagens de reconhecimento aos colaboradores."""
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y

        self.setWindowTitle("Agradecimentos")
        self.setGeometry(int(460 * self.scale_x), int(320 * self.scale_y), int(800 * self.scale_x), int(450 * self.scale_y))

        agradecimentos_texto = (
        "<h2 style='text-align: center; margin-bottom: {int(20 * scale_y)}px;'>Agradecimentos</h2>"
        "<p style='text-align: center; font-size: {int(16 * scale_y)}px; margin-top: {int(20 * scale_y)}px;'>"
        "Este projeto é o resultado de esforços colaborativos e dedicação de muitas pessoas e instituições. "
        "Gostaríamos de expressar nossa mais sincera gratidão aos seguintes contribuintes e apoiadores:"
        "</p>"
        "<ul style='font-size: {int(14 * scale_y)}px; margin-left: {int(40 * scale_x)}px;'>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "À equipe de desenvolvimento, pelo comprometimento e empenho em todas as etapas deste projeto."
        "</li>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "Aos professores e colegas, cujas ideias e feedbacks enriqueceram significativamente o trabalho."
        "</li>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "A você, usuário, por confiar e utilizar este programa. Seu apoio nos motiva a seguir inovando."
        "</li>"
        "</ul>"
        "<p style='text-align: center; font-size: {int(16 * scale_y)}px; margin-top: {int(20 * scale_y)}px;'>"
        "Reconhecemos, com especial gratidão, o apoio de:"
        "</p>"
        "<ul style='font-size: {int(14 * scale_y)}px; margin-left: {int(40 * scale_x)}px;'>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "CNPq (Conselho Nacional de Desenvolvimento Científico e Tecnológico), pelo suporte financeiro que possibilitou este trabalho."
        "</li>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "InfantId, por sua parceria e suporte contínuo durante o desenvolvimento deste projeto."
        "</li>"
        "<li style='margin-bottom: {int(10 * scale_y)}px;'>"
        "UTFPR campus Pato Branco, por oferecer a infraestrutura necessária e o apoio institucional essencial."
        "</li>"
        "</ul>"
        "<p style='text-align: center; font-size: {int(16 * scale_y)}px; margin-top: {int(20 * scale_y)}px;'>"
        "Estamos comprometidos com a melhoria contínua deste programa e valorizamos seu feedback para tornar nossa ferramenta ainda melhor."
        "</p>"
        "<p style='text-align: center; font-size: {int(16 * scale_y)}px; margin-top: {int(20 * scale_y)}px;'>"
        "Desenvolvido por:"
        "</p>"
        "<h3 style='text-align: center; margin-top: {int(10 * scale_y)}px;'>Matheus Augusto</h3>"
        "<h4 style='text-align: center; margin-top: {int(10 * scale_y)}px;'>"
        "Email: matheusaugustooliveira@alunos.utfpr.edu.br"
        "</h4>"
        "<p style='text-align: center; font-size: {int(14 * scale_y)}px;'>"
        "<br>Obrigado por fazer parte desta jornada! <br>"
        "</p>"
    )

        text_edit = QTextEdit(self)
        font = QFont("Times", int(14 * scale_x)) 
        text_edit.setFont(font)
        text_edit.setHtml(agradecimentos_texto)
        text_edit.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(text_edit)

        close_button = QPushButton("Fechar", self)
        close_button.clicked.connect(self.accept)
        close_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) 
        layout.addWidget(close_button)

class ManualDialog(QDialog):
    """Classe que exibe um manual do usuário, fornecendo instruções sobre como utilizar a interface e as funcionalidades do aplicativo."""
    def __init__(self, parent, scale_x, scale_y):
        """Inicializa a janela do manual do usuário, definindo a interface gráfica e carregando o conteúdo explicativo."""
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y

        self.setWindowTitle("Manual do Usuário")
        self.setGeometry(int(485 * self.scale_x), int(200 * self.scale_y), int(700 * self.scale_x), int(650 * self.scale_y))

        text_edit = QTextEdit(self)
        
        font = QFont("Times", int(14 * scale_x)) 
        text_edit.setFont(font)  

        manual_texto = (
            "<h2 style='text-align: center;'>Bem-vindo ao programa de alinhamento de imagens!</h2>"  
            "<p style='text-align: center; margin-bottom: {int(14 * scale_y)}px;'>Aqui estão algumas dicas para você começar:</p>" 
            "<br>"
            "<ol>" 
            "<li><b>Seleção das Imagens</b>: As imagens devem estar sem fundo. Utilize ferramentas adequadas para preparar as imagens antes de carregá-las no programa.</li>"
            "<br>"
            "<li><b>Carregar Imagem Base</b>: Clique em 'Carregar Imagem Base' para escolher a imagem que será usada como fundo.</li>"
            "<br>"
            "<li><b>Carregar Imagem Sobreposta</b>: Clique em 'Carregar Imagem Sobreposta' para adicionar uma nova imagem sobreposta ao fundo.</li>"
            "<br>"
            "<li><b>Mover Imagem Sobreposta</b>: Clique e arraste a imagem sobreposta para movê-la para a posição desejada em relação à imagem base.</li>"
            "<br>"
            "<li><b>Rotacionar Imagem Sobreposta</b>: Clique com o botão direito do mouse e mova o mouse para girar a imagem sobreposta, ajustando sua orientação em relação à imagem base.</li>"
            "<br>"
            "<li><b>Ajuste Fino</b>: Utilize os campos de texto para ajustar a posição e a rotação da imagem sobreposta com valores numéricos. Isso permite ajustes precisos na largura, altura e posição da imagem.</li>"
            "<br>"
            "<li><b>Incrementar/Decrementar Altura/Largura</b>: Use os botões de incremento e decremento para ajustar a altura e a largura da imagem sobreposta em incrementos de 1 pixel.</li>"
            "<br>"
            "<li><b>Zoom</b>: Aplique zoom na imagem para visualização mais detalhada ou para ajustar o nível de zoom durante o processo de alinhamento. O zoom pode ser resetado a qualquer momento para o valor padrão.</li>"
            "<br>"
            "<li><b>Alinhamento</b>: O programa calcula a sobreposição entre a imagem base e a imagem sobreposta. A interseção entre as imagens é calculada, e a porcentagem de sobreposição é exibida no campo de alinhamento.</li>"
            "<br>"
            "<li><b>Mudar Cor de Fundo</b>: Você pode alterar a cor de fundo da área de visualização para facilitar a visualização das imagens e destacar a imagem sobreposta.</li>"
            "<br>"
            "<li><b>Mudar Cor da Imagem Sobreposta</b>: A cor da imagem sobreposta pode ser alterada para um tom específico, facilitando a visualização da sobreposição em relação à imagem base.</li>"
            "<br>"
            "<li><b>Salvar Imagem</b>: Após ajustar a imagem sobreposta à imagem base, clique em 'Salvar Imagem' para salvar o resultado final do alinhamento em um arquivo de imagem.</li>"
            "<br>"
            "<li><b>Realizar Ajuste Fino de Posição e Tamanho</b>: Para um alinhamento mais preciso, utilize os campos de altura e largura ou os botões de incremento para realizar ajustes finos.</li>"
            "<br>"  
            "<li><b>Log</b>: Clique em 'Log' para abrir a janela de reaplicação automática. Em seguida, clique em 'Carregar Log' e selecione o arquivo JSON salvo anteriormente. Depois, clique em 'Confirmar'. O programa aplicará automaticamente as configurações de alinhamento salvas e exibirá uma mensagem ao final do processo.</li>"
            "</ol>"
            "<br>"
            "<br>"
            "<p style='text-align: center; margin-top: {int(14 * scale_y)}px;'>Esse manual fornece um guia rápido de como usar as funcionalidades principais do programa. Caso tenha dúvidas adicionais, entre em contato com:  <br> </p>"  
            "<h3 style='text-align: center; margin-top: {int(14 * scale_y)}px; '>Matheus Augusto </h3>" 
            "<h3 style='text-align: center; margin-top: {int(14 * scale_y)}px; '>Email: matheusaugustooliveira@alunos.utfpr.edu.br <br></h3>"  
        )

        text_edit.setHtml(manual_texto)  
        text_edit.setReadOnly(True)  

        layout = QVBoxLayout(self)
        layout.addWidget(text_edit)

        close_button = QPushButton("Fechar", self)
        close_button.clicked.connect(self.accept)  
        layout.addWidget(close_button)

class CorDialog(QDialog):
    """ Janela de seleção de cores para a interface do aplicativo. Permite ao usuário modificar a cor de fundo, a cor da imagem base e a cor da imagem sobreposta, garantindo personalização visual do ambiente de edição."""
    def __init__(self, parent, scale_x=1.0, scale_y=1.0):
        """Inicializa a janela de seleção de cores, configurando os botões para alterar a cor de fundo, base e sobreposta."""
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y

        self.setWindowTitle("Opções para Colorir")
        self.setGeometry(int(15 * self.scale_x), int(65 * self.scale_y), int(250 * self.scale_x), int(425 * self.scale_y))
        
        self.btn_fundo = QPushButton("Mudar Cor de Fundo", self)
        self.btn_fundo.setGeometry(int(25 * self.scale_x), int(15 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_fundo.clicked.connect(self.mudar_cor_fundo)

        self.btn_base = QPushButton("Mudar Cor da Base", self)
        self.btn_base.setGeometry(int(25 * self.scale_x), int(55 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_base.clicked.connect(self.mudar_cor_base)

        self.btn_sobreposta = QPushButton("Mudar Cor da Sobreposta", self)
        self.btn_sobreposta.setGeometry(int(25 * self.scale_x), int(95 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_sobreposta.clicked.connect(self.mudar_cor_sobreposta)

        self.btn_cor_minu = QPushButton("Mudar Cor das Minúcias", self)
        self.btn_cor_minu.setGeometry(int(25 * self.scale_x), int(135 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_minu.clicked.connect(self.mudar_cor_minucias)

        self.btn_cor_delt = QPushButton("Mudar Cor do Delta", self)
        self.btn_cor_delt.setGeometry(int(25 * self.scale_x), int(175 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_delt.clicked.connect(self.mudar_cor_delta)

        self.btn_cor_core = QPushButton("Mudar Cor do Core", self)
        self.btn_cor_core.setGeometry(int(25 * self.scale_x), int(215 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_core.clicked.connect(self.mudar_cor_core)

        self.btn_cor_marc = QPushButton("Mudar Cor da Marcação", self)
        self.btn_cor_marc.setGeometry(int(25 * self.scale_x), int(255 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_marc.clicked.connect(self.mudar_cor_marcacao)

        self.btn_cor_dir = QPushButton("Mudar Cor da Direção", self)
        self.btn_cor_dir.setGeometry(int(25 * self.scale_x), int(295 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_dir.clicked.connect(self.mudar_cor_direcao)

        self.btn_cor_graf = QPushButton("Mudar Cor Grafo", self)
        self.btn_cor_graf.setGeometry(int(25 * self.scale_x), int(335 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_graf.clicked.connect(self.mudar_cor_grafo) 

        self.btn_cor_mat = QPushButton("Mudar Cor do Matching", self)
        self.btn_cor_mat.setGeometry(int(25 * self.scale_x), int(375 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))
        self.btn_cor_mat.clicked.connect(self.mudar_cor_matching)

    def mudar_cor_fundo(self):
        """Abre um diálogo para escolher a cor de fundo e aplica a nova cor à interface."""
        cor = escolher_cor(self, self.parent().rect_color)
        if cor.isValid():
            self.parent().rect_color = cor
            self.parent().update()

            self.parent().settings.setValue("cor_fundo", cor.name())
            self.parent().settings.sync()
            self.parent().carregar_cor_fundo()

            if cor == self.parent().easter_egg_color:
                self.parent().activate_easter_egg()
            else:
                self.parent().deactivate_easter_egg()

        QApplication.instance().setStyleSheet(stylesheet)

    def mudar_cor_base(self):
        """Abre um diálogo para escolher uma nova cor para a imagem base e aplica a mudança."""
        parent = self.parent() 
        if parent.base_scaled:  
            cor = escolher_cor(self, parent.rect_color)
            if cor.isValid():
                colored_pixmap = self.aplicar_cor_na_imagem(parent.base_scaled, cor)
                
                parent.base_colored = colored_pixmap.copy()  
                parent.base_scaled = parent.base_colored.copy()

                parent.salvar_cor_aplicada(cor, "base")  
                parent.update()
            else:
                QMessageBox.warning(self, "Atenção", "A cor selecionada é inválida.")
        else:
            QMessageBox.warning(self, "Atenção", "A imagem base não está carregada.")

    def mudar_cor_sobreposta(self):
        """Abre um diálogo para escolher uma nova cor para a imagem sobreposta e aplica a mudança."""
        parent = self.parent()  
        if parent.sobreposta_scaled:  
            cor = escolher_cor(self, parent.rect_color)
            if cor.isValid():
                colored_pixmap = self.aplicar_cor_na_imagem(parent.sobreposta_scaled, cor)

                parent.sobreposta_colored = colored_pixmap.copy()  
                parent.sobreposta_scaled = parent.sobreposta_colored.copy()

                parent.salvar_cor_aplicada(cor, "sobreposta")   
                
                parent.cor_sobreposta_alterada = True  
                parent.update()
            else:
                QMessageBox.warning(self, "Atenção", "A cor selecionada é inválida.")
        else:
            QMessageBox.warning(self, "Atenção", "A imagem sobreposta não está carregada.")

    def mudar_cor_minucias(self):
        """Permite escolher nova cor para as minúcias e reaplica se já existirem."""
        parent = self.parent()
        cor_atual = QColor(*[int(c) for c in reversed(parent.cor_minucias)])
        cor = escolher_cor(self, cor_atual)

        if cor.isValid():
            parent.cor_minucias = (cor.blue(), cor.green(), cor.red())

            parent.settings.setValue("cor_minucias", (cor.red(), cor.green(), cor.blue()))
            parent.settings.sync()

            if hasattr(parent, 'minucias_pixmap') and parent.minucias_pixmap:
                parent.redesenhar_minucias()

    def mudar_cor_core(self):
        cor = escolher_cor(self, self.parent().cor_core)
        if cor.isValid():
            self.parent().cor_core = cor
            self.parent().settings.setValue("cor_core", cor.name())
            self.parent().settings.sync()
            self.parent().update()

    def mudar_cor_delta(self):
        cor = escolher_cor(self, self.parent().cor_delta)
        if cor.isValid():
            self.parent().cor_delta = cor
            self.parent().settings.setValue("cor_delta", cor.name())
            self.parent().settings.sync()
            self.parent().update()

    def mudar_cor_marcacao(self):
        """Abre a paleta de cores e altera a cor da marcação (números)"""
        parent = self.parent()
        cor_atual = parent.cor_marcacao
        cor = escolher_cor(self, cor_atual)

        if cor.isValid():
            parent.cor_marcacao = cor
            parent.settings.setValue("cor_marcacao", cor.name())
            parent.settings.sync()
            parent.update()

    def mudar_cor_grafo(self):
        """Permite mudar a cor das linhas do grafo."""
        cor = escolher_cor(self, self.parent().cor_grafo)
        if cor.isValid():
            self.parent().cor_grafo = cor
            self.parent().settings.setValue("cor_grafo", cor.name())
            self.parent().settings.sync()
            self.parent().update()
    
    def mudar_cor_direcao(self):
        cor = escolher_cor(self, self.parent().cor_direcao_minucias)
        if cor.isValid():
            self.parent().cor_direcao_minucias = cor
            self.parent().settings.setValue("cor_direcao_minucias", cor.name())
            self.parent().settings.sync()
            self.parent().update()

    def mudar_cor_matching(self):
        cor = escolher_cor(self, self.parent().cor_matching)
        if cor.isValid():
            self.parent().cor_matching = cor
            self.parent().settings.setValue("cor_matching", cor.name())
            self.parent().settings.sync()
            self.parent().update()
            
    def aplicar_cor_na_imagem(self, pixmap, cor):
        """Aplica a cor escolhida a uma imagem (QPixmap), preservando a transparência."""
        if pixmap is None or pixmap.isNull():
            return QPixmap()

        result = QPixmap(pixmap.size())
        result.fill(Qt.GlobalColor.transparent)

        painter = QPainter(result)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), cor)
        painter.end()

        return result
       
class Carregar(QDialog):
    """Janela de seleção de imagens para carregar base e sobreposta."""
    def __init__(self, parent, scale_x, scale_y):
        """Inicializa a janela de carregamento de imagens. Configura a escala da interface gráfica e prepara os elementos da janela de diálogo."""
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y

        self.setWindowTitle("Opções para Carregar Imagens")
        self.setGeometry(int(15 * scale_x), int(550 * scale_y), int(300 * scale_x), int(125 * scale_y))

        self.check_base_ok = QCheckBox("", self)
        self.check_base_ok.setGeometry(int(10 * self.scale_x), int(25 * self.scale_y), int(100 * self.scale_x), int(30 * self.scale_y))
        self.check_base_ok.setEnabled(False)

        self.btn_base = QPushButton("Carregar Imagem Base", self)
        self.btn_base.setGeometry(int(35 * scale_x), int(20 * scale_y), int(215 * scale_x), int(35 * scale_y))
        self.btn_base.clicked.connect(parent.carregar_imagem_base)

        self.btn_excluir_base = QPushButton(self)
        self.btn_excluir_base.setGeometry(self.btn_base.x() + self.btn_base.width() + int(10 * scale_x), self.btn_base.y(), int(25 * scale_x), int(35 * scale_y))
        self.btn_excluir_base.setIcon(QIcon.fromTheme("user-trash"))
        self.btn_excluir_base.clicked.connect(parent.excluir_imagem_base)

        self.check_sobreposta_imagem1 = QCheckBox("", self)
        self.check_sobreposta_imagem1.setGeometry(int(10 * self.scale_x), int(75 * self.scale_y), int(100 * self.scale_x), int(30 * self.scale_y))
        self.check_sobreposta_imagem1.setEnabled(False) 

        self.btn_sobreposta = QPushButton("Carregar Imagem Sobreposta", self)
        self.btn_sobreposta.setGeometry(int(35 * scale_x), int(70 * scale_y), int(215 * scale_x), int(35 * scale_y))
        self.btn_sobreposta.clicked.connect(parent.carregar_imagem_sobreposta)

        self.btn_excluir_sobreposta = QPushButton(self)
        self.btn_excluir_sobreposta.setGeometry(self.btn_sobreposta.x() + self.btn_sobreposta.width() + int(10 * scale_x), self.btn_sobreposta.y(), int(25 * scale_x), int(35 * scale_y))
        self.btn_excluir_sobreposta.setIcon(QIcon.fromTheme("user-trash"))
        self.btn_excluir_sobreposta.clicked.connect(parent.excluir_imagem_sobreposta)

        parent.imagem_base_ok.connect(self.atualizar_checkbox_base)
        parent.imagem_sobreposta_carregada.connect(self.atualizar_checkbox_sobreposta)

        checkbox_base_estado = parent.settings.value("checkbox_base", False, type=bool)
        self.check_base_ok.setChecked(checkbox_base_estado)

        if checkbox_base_estado:
            self.check_base_ok.setStyleSheet("""
                QCheckBox::indicator:checked {
                    background-color: cyan;  
                    border: 2px solid cyan;
                }
            """)

        checkbox_sobreposta_estado = parent.settings.value("checkbox_sobreposta", False, type=bool)
        self.check_sobreposta_imagem1.setChecked(checkbox_sobreposta_estado)

        if checkbox_sobreposta_estado:
            self.check_sobreposta_imagem1.setStyleSheet("""
                QCheckBox::indicator:checked {
                    background-color: cyan;  
                    border: 2px solid cyan;
                }
            """)

    def atualizar_checkbox_base(self, carregada):
        """Atualiza a checkbox de imagem base quando a imagem é carregada ou deletada, alterando sua cor e borda."""
        self.check_base_ok.setChecked(carregada)

        if carregada:
            self.check_base_ok.setStyleSheet("""
                QCheckBox::indicator:checked {
                    background-color: cyan;  /* Cor do fundo quando marcado */
                    border: 2px solid cyan;  /* Cor da borda ao marcar */
                }
            """)

    def atualizar_checkbox_sobreposta(self, carregada):
        """Atualiza a checkbox de imagem sobreposta quando a imagem é carregada ou deletada, alterando sua cor e borda."""
        self.check_sobreposta_imagem1.setChecked(carregada)

        if carregada:
            self.check_sobreposta_imagem1.setStyleSheet("""
                QCheckBox::indicator:checked {
                    background-color: cyan;  /* Cor do fundo quando marcado */
                    border: 2px solid cyan;  /* Cor da borda ao marcar */
                }
            """)

class SelecaoDialog(QDialog):
    """ Janela de seleção para escolher um tipo de impressão digital. Exibe uma interface com botões representando diferentes padrões biométricos para o usuário selecionar."""

    def __init__(self, parent=None, scale_x=1.0, scale_y=1.0):
        """Inicializa a janela de seleção de tipos de impressão digital, exibindo botões para escolha do usuário."""
        super().__init__(parent)

        self.setWindowTitle("Escolha um Tipo de Impressão Digital")
        self.setGeometry(int(160 * scale_x), int(35 * scale_y), int(1100 * scale_x), int(100 * scale_y))

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint) 
        self.setWindowModality(Qt.WindowModality.ApplicationModal)  

        self.scale_x = scale_x
        self.scale_y = scale_y

        layout = QHBoxLayout()

        self.tipos = [
            ("Laco Ulnar", "Imagens/Laço Ulnar.png"),
            ("Laco Radial", "Imagens/Laço Radial.png"),
            ("Arco Simples", "Imagens/Arco Simples.png"),
            ("Arco Tendido", "Imagens/Arco Tendido.png"),
            ("Espiral Concentrica", "Imagens/Espiral Concêntrica.png"),
            ("Espiral", "Imagens/Espiral.png"),
            ("Espiral de Pressao", "Imagens/Espiral de Pressão.png"),
            ("Espiral Implodente", "Imagens/Espiral Implodente.png"),
            ("Espiral Composta", "Imagens/Espiral Composta.png"),
            ("Olho de Pavao", "Imagens/Olho de Pavão.png"),
            ("Padrao Variavel", "Imagens/Padrão Variável.png"),
        ]

        for nome, imagem in self.tipos:
            btn = QPushButton()
            btn.setIcon(QIcon(resource_path(imagem)))
            btn.setIconSize(QSize(int(100 * self.scale_x), int(100 * self.scale_y)))  
            btn.setToolTip(nome)  
            btn.clicked.connect(lambda _, n=nome: self.selecionar_nome(n))
            layout.addWidget(btn)

        self.setLayout(layout)
        
    def selecionar_nome(self, nome):
        """Registra o nome da opção selecionada e fecha o diálogo."""
        self.nome_selecionado = nome
        self.accept()

    def keyPressEvent(self, event):
        """Impede que a tecla ESC ou Ctrl + Esc fechem ou alterem o software."""
        if event.key() == Qt.Key.Key_Escape and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            print("Ctrl + Esc bloqueado!")
            return  

        elif event.key() == Qt.Key.Key_Escape:
            event.ignore()  

        else:
            super().keyPressEvent(event)  

    def closeEvent(self, event):
        """Impede que a janela seja fechada manualmente."""
        event.ignore()  
        
class JanelaLogDialog(QDialog):
    confirmar_aplicacao = pyqtSignal(str)  

    def __init__(self, parent, scale_x, scale_y):
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y
        
        self.setWindowTitle("Aplicar Log Automático")
        self.setGeometry(int(15 * scale_x), int(730 * scale_y), int(250 * scale_x), int(110 * scale_y))
        
        self.btn_selecionar_arquivo = QPushButton("Selecionar ZIP", self)
        self.btn_selecionar_arquivo.setGeometry(int(25 * scale_x), int(20 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_selecionar_arquivo.clicked.connect(self.selecionar_arquivo)
        
        self.btn_confirmar = QPushButton("Confirmar e Aplicar", self)
        self.btn_confirmar.setGeometry(int(25 * scale_x), int(60 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_confirmar.clicked.connect(self.enviar_confirmacao)
        
        self.arquivo_log_path = None

    def selecionar_arquivo(self):
        file_path = escolher_arquivo(self, "Selecionar ZIP com log", "Arquivos ZIP (*.zip);;JSON Files (*.json)")
        if file_path:
            self.arquivo_log_path = file_path
    
    def enviar_confirmacao(self):
        if self.arquivo_log_path:
            self.confirmar_aplicacao.emit(self.arquivo_log_path)
            self.close()
        else:
            QMessageBox.warning(self, "Erro", "Nenhum arquivo ZIP selecionado.")

class TransparenciaDialog(QDialog):
    """Janela de controle de transparência para as imagens base e sobreposta."""
    def __init__(self, parent, scale_x, scale_y):
        """Inicializa a janela de controle de transparência, configurando os elementos da interface."""
        super().__init__(parent)

        self.scale_x = scale_x
        self.scale_y = scale_y
        self.parent = parent 

        self.setWindowTitle("Controles de Transparência")
        self.setGeometry(int(1415 * scale_x), int(515 * scale_y), int(250 * scale_x), int(205 * scale_y))

        self.label3 = QLabel(f"Transparência: {int(parent.transparencia * 100)}%", self)
        self.label3.setGeometry(int(25 * self.scale_x), int(10 * self.scale_y), int(200 * self.scale_x), int(30 * self.scale_y))

        self.slider3 = QSlider(Qt.Orientation.Horizontal, self)  
        self.slider3.setGeometry(int(25 * self.scale_x), int(35 * self.scale_y), int(200 * self.scale_x), int(20 * self.scale_y))
        self.slider3.setRange(0, 100)
        self.slider3.setValue(int(parent.transparencia * 100))  
        self.slider3.valueChanged.connect(self.atualizar_transparencia)

        self.check_base_imagem = QCheckBox("Base Transparente", self)
        self.check_base_imagem.setGeometry(int(25 * self.scale_x), int(65 * self.scale_y), int(210 * self.scale_x), int(30 * self.scale_y))
        self.check_base_imagem.setChecked(parent.check_base_imagem.isChecked())
        self.check_base_imagem.stateChanged.connect(self.alterar_transparencia_imagem_base)
        self.check_base_imagem.setEnabled(parent.check_base_imagem.isEnabled())

        self.check_imagem_sobreposta = QCheckBox("Sobreposta Transparente", self)
        self.check_imagem_sobreposta.setGeometry(int(25 * self.scale_x), int(95 * self.scale_y), int(210 * self.scale_x), int(30 * self.scale_y))
        self.check_imagem_sobreposta.setChecked(parent.check_imagem_sobreposta.isChecked())
        self.check_imagem_sobreposta.stateChanged.connect(self.alterar_transparencia_imagem_sobreposta)
        self.check_imagem_sobreposta.setEnabled(parent.check_imagem_sobreposta.isEnabled())

        self.check_minucias = QCheckBox("Minúcias Transparentes", self)
        self.check_minucias.setGeometry(int(25 * self.scale_x), int(125 * self.scale_y), int(210 * self.scale_x), int(30 * self.scale_y))
        self.check_minucias.setEnabled(hasattr(parent, 'minucias_scaled') and parent.minucias_scaled is not None)
        self.check_minucias.setChecked(parent.check_minucias.isChecked() if hasattr(parent, 'check_minucias') else False)
        self.check_minucias.stateChanged.connect(self.alterar_transparencia_minucias)

        self.check_numeros_marcacao = QCheckBox("Exibir Números da Marcação", self)
        self.check_numeros_marcacao.setGeometry(int(25 * self.scale_x), int(155 * self.scale_y), int(210 * self.scale_x), int(30 * self.scale_y))
        valor_salvo = self.parent.settings.value("mostrar_numeros_marcacao", True, type=bool)
        self.check_numeros_marcacao.setChecked(False)
        self.check_numeros_marcacao.setEnabled(False)
        self.atualizar_estado_marcacao()
        self.check_numeros_marcacao.stateChanged.connect(self.toggle_numeros_marcacao)

    def alterar_transparencia_imagem_base(self, estado):
        """Sincroniza o estado do checkbox com o parent e aciona a função correspondente."""
        self.parent.check_base_imagem.setChecked(self.check_base_imagem.isChecked())
        self.parent.alterar_transparencia_imagem_base()

    def alterar_transparencia_imagem_sobreposta(self, estado):
        """Sincroniza o estado do checkbox com o parent e aciona a função correspondente."""
        self.parent.check_imagem_sobreposta.setChecked(self.check_imagem_sobreposta.isChecked())
        self.parent.alterar_transparencia_imagem_sobreposta()

    def atualizar_transparencia(self, valor):
        """Atualiza o valor de transparência no parent (Layout) e aciona a atualização visual."""
        self.parent.transparencia = valor / 100.0
        self.label3.setText(f"Transparência: {valor}%")  
        self.parent.update()

    def alterar_transparencia_minucias(self, estado):
        """Sincroniza o estado do checkbox com o parent e aciona a função correspondente."""
        if hasattr(self.parent, 'check_minucias'):
            self.parent.check_minucias.setChecked(self.check_minucias.isChecked())
            self.parent.alterar_transparencia_minucias()
    
    def atualizar_estado_minucias(self):
        """Atualiza o estado do checkbox de minúcias de acordo com o estado atual no parent."""
        self.check_minucias.setEnabled(hasattr(self.parent, 'minucias_scaled') and self.parent.minucias_scaled is not None)
        if hasattr(self.parent, 'check_minucias'):
            self.check_minucias.setChecked(self.parent.check_minucias.isChecked())

    def toggle_numeros_marcacao(self, estado):
        """Liga ou desliga a exibição dos números das minúcias"""
        self.parent.settings.setValue("mostrar_numeros_marcacao", bool(estado))
        self.parent.mostrar_numeros_marcacao = bool(estado)
        self.parent.update()

    def atualizar_estado_marcacao(self):
        """Ativa ou desativa o checkbox de exibição dos números, com base na existência das minúcias."""
        if hasattr(self.parent, 'minutiae_points') and self.parent.minutiae_points:
            self.check_numeros_marcacao.setEnabled(True)

            valor_salvo = self.parent.settings.value("mostrar_numeros_marcacao", True, type=bool)
            self.check_numeros_marcacao.setChecked(valor_salvo)
        else:
            self.check_numeros_marcacao.setChecked(False)
            self.check_numeros_marcacao.setEnabled(False)

    def atualizar_estado_marcacao(self):
        """Ativa ou desativa o checkbox de exibição dos números, com base na existência das minúcias."""
        if hasattr(self.parent, 'minutiae_points') and self.parent.minutiae_points:
            self.check_numeros_marcacao.setEnabled(True)
            valor_salvo = self.parent.settings.value("mostrar_numeros_marcacao", True, type=bool)
            self.check_numeros_marcacao.setChecked(valor_salvo)
        else:
            self.check_numeros_marcacao.setChecked(False)
            self.check_numeros_marcacao.setEnabled(False)

class MarcarDialog(QDialog):
    def __init__(self, parent, scale_x, scale_y):
        super().__init__(parent)
        self.setWindowTitle("Detectar e Marcar Pontos")
        self.setGeometry(int(1415 * scale_x), int(65 * scale_y), int(250 * scale_x), int(360 * scale_y))

        self.parent = parent

        self.label1 = QLabel("Quantidade de Minucias:", self)
        self.label1.move(int(25 * scale_x), int(10 * scale_y))

        self.campo1 = QLineEdit(self)
        self.campo1.setGeometry(int(25 * scale_x), int(35 * scale_y), int(170 * scale_x), int(30 * scale_y))
        self.campo1.setText("25")

        self.botao_up_y = QToolButton(self)
        self.botao_up_y.setText("▲")
        self.botao_up_y.setGeometry(int(195 * scale_x), int(35 * scale_y), int(30 * scale_x), int(15 * scale_y))
        self.botao_up_y.clicked.connect(self.aumentar_quantidade)

        self.botao_down_y = QToolButton(self)
        self.botao_down_y.setText("▼")
        self.botao_down_y.setGeometry(int(195 * scale_x), int(50 * scale_y), int(30 * scale_x), int(15 * scale_y))
        self.botao_down_y.clicked.connect(self.diminuir_quantidade)

        self.label0 = QLabel("Distância entre Minucias:", self)
        self.label0.move(int(25 * scale_x), int(80 * scale_y))

        self.campo0 = QLineEdit(self)
        self.campo0.setGeometry(int(25 * scale_x), int(105 * scale_y), int(170 * scale_x), int(30 * scale_y))
        self.campo0.setText("60")

        self.botao_up_x = QToolButton(self)
        self.botao_up_x.setText("▲")
        self.botao_up_x.setGeometry(int(195 * scale_x), int(105 * scale_y), int(30 * scale_x), int(15 * scale_y))
        self.botao_up_x.clicked.connect(self.aumentar_distancia)

        self.botao_down_x = QToolButton(self)
        self.botao_down_x.setText("▼")
        self.botao_down_x.setGeometry(int(195 * scale_x), int(120 * scale_y), int(30 * scale_x), int(15 * scale_y))
        self.botao_down_x.clicked.connect(self.diminuir_distancia)

        self.btn_detectar = QPushButton("Detectar Pontos", self)
        self.btn_detectar.setGeometry(int(25 * scale_x), int(150 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_detectar.clicked.connect(self.detectar_minucias)

        self.btn_marcar_delta = QPushButton("Delta", self)
        self.btn_marcar_delta.setGeometry(int(25 * scale_x), int(190 * scale_y), int(130 * scale_x), int(30 * scale_y))
        self.btn_marcar_delta.clicked.connect(lambda: self.definir_modo("delta"))

        self.btn_editar_delta = QPushButton(self)
        icone_editar = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
        self.btn_editar_delta.setIcon(icone_editar)
        self.btn_editar_delta.setGeometry(int(160 * scale_x), int(190 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_editar_delta.clicked.connect(lambda: self.definir_modo_editar("delta"))

        self.btn_apagar_delta = QPushButton(self)
        self.btn_apagar_delta.setIcon(QIcon.fromTheme("edit-delete"))
        self.btn_apagar_delta.setGeometry(int(195 * scale_x), int(190 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_apagar_delta.clicked.connect(lambda: self.definir_modo_apagar("delta"))

        self.btn_marcar_core = QPushButton("Core", self)
        self.btn_marcar_core.setGeometry(int(25 * scale_x), int(230 * scale_y), int(130 * scale_x), int(30 * scale_y))
        self.btn_marcar_core.clicked.connect(lambda: self.definir_modo("core"))

        self.btn_editar_core = QPushButton(self)
        self.btn_editar_core.setIcon(icone_editar)
        self.btn_editar_core.setGeometry(int(160 * scale_x), int(230 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_editar_core.clicked.connect(lambda: self.definir_modo_editar("core"))

        self.btn_apagar_core = QPushButton(self)
        self.btn_apagar_core.setIcon(QIcon.fromTheme("edit-delete"))
        self.btn_apagar_core.setGeometry(int(195 * scale_x), int(230 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_apagar_core.clicked.connect(lambda: self.definir_modo_apagar("core"))

        self.btn_marcar_minucia = QPushButton("Minúcia", self)
        self.btn_marcar_minucia.setGeometry(int(25 * scale_x), int(270 * scale_y), int(130 * scale_x), int(30 * scale_y))
        self.btn_marcar_minucia.clicked.connect(lambda: self.definir_modo("minucia"))

        self.btn_editar_minucia = QPushButton(self)
        icone = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
        self.btn_editar_minucia.setIcon(icone)
        self.btn_editar_minucia.setGeometry(int(160 * scale_x), int(270 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_editar_minucia.clicked.connect(lambda: self.definir_modo_editar("minucia"))

        self.btn_apagar_minucia = QPushButton(self)
        self.btn_apagar_minucia.setIcon(QIcon.fromTheme("edit-delete"))
        self.btn_apagar_minucia.setGeometry(int(195 * scale_x), int(270 * scale_y), int(30 * scale_x), int(30 * scale_y))
        self.btn_apagar_minucia.clicked.connect(lambda: self.definir_modo_apagar("minucia"))

        self.btn_parar_marcacao = QPushButton("Marcado", self)
        self.btn_parar_marcacao.setGeometry(int(25 * scale_x), int(310 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_parar_marcacao.clicked.connect(self.cancelar_modo_marcacao)

        self.modo_operacao = None

    def aumentar_quantidade(self):
        try:
            valor = int(self.campo1.text())
            valor += 1
            self.campo1.setText(str(valor))
        except ValueError:
            pass

    def diminuir_quantidade(self):
        try:
            valor = int(self.campo1.text())
            if valor > 1:
                valor -= 1
                self.campo1.setText(str(valor))
        except ValueError:
            pass

    def aumentar_distancia(self):
        try:
            valor = int(self.campo0.text())
            valor += 1
            self.campo0.setText(str(valor))
        except ValueError:
            pass

    def diminuir_distancia(self):
        try:
            valor = int(self.campo0.text())
            if valor > 1:
                valor -= 1
                self.campo0.setText(str(valor))
        except ValueError:
            pass

    def detectar_minucias(self):
        try:
            quantidade = int(self.campo1.text())
            distancia = int(self.campo0.text())
        except ValueError:
            QMessageBox.warning(self, "Erro", "Por favor, insira valores numéricos válidos para quantidade e distância.")
            return

        if quantidade <= 0 or distancia <= 0:
            QMessageBox.warning(self, "Erro", "Quantidade e distância devem ser valores positivos.")
            return

        self.parent.quantidade_minucias = quantidade
        self.parent.distancia_minima = distancia

        self.parent.detectar_minutiae()

    def definir_modo(self, tipo):
        self.modo_operacao = f"marcar_{tipo}"
        self.parent.modo_marcacao = self.modo_operacao  

    def definir_modo_editar(self, tipo):
        self.modo_operacao = f"editar_{tipo}"
        self.parent.modo_marcacao = self.modo_operacao

    def definir_modo_apagar(self, tipo):
        self.modo_operacao = f"apagar_{tipo}"
        self.parent.modo_marcacao = self.modo_operacao

    def cancelar_modo_marcacao(self):
        """Cancela qualquer modo de marcação/edição/apagar ativo."""
        self.modo_operacao = None
        self.parent.modo_marcacao = None

class GrafoDialog(QDialog):
    """Janela de configurações para o grafo, permitindo ajustar a espessura das linhase gerar o grafo com base nas minúcias detectadas."""
    def __init__(self, parent, scale_x, scale_y):
        """Inicializa a janela de configuração do grafo com controles de espessura e geração."""
        super().__init__(parent)
        self.setWindowTitle("Configurações do Grafo")
        self.setGeometry(int(1415 * scale_x), int(810 * scale_y), int(250 * scale_x), int(230 * scale_y))
        self.parent = parent

        self.label = QLabel(f"Espessura da Linha: {parent.espessura_grafo}", self)
        self.label.setGeometry(int(25 * scale_x), int(10 * scale_y), int(200 * scale_x), int(30 * scale_y))

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setGeometry(int(25 * scale_x), int(40 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.slider.setRange(1, 10)
        self.slider.setValue(parent.espessura_grafo)
        self.slider.valueChanged.connect(self.ajustar_espessura)

        self.btn_grafo = QPushButton("Gerar Grafo", self)
        self.btn_grafo.setGeometry(int(25 * scale_x), int(80 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_grafo.clicked.connect(self.tentar_gerar_grafo)

        self.check_grafo = QCheckBox("Exibir Grafo", self)
        self.check_grafo.setGeometry(int(25 * scale_x), int(120 * scale_y), int(210 * scale_x), int(30 * scale_y))
        self.check_grafo.setChecked(False)
        self.check_grafo.setEnabled(False)
        self.check_grafo.stateChanged.connect(self.parent.update)

        self.tipo_label = QLabel("Tipo de Grafo:", self)
        self.tipo_label.setGeometry(int(25 * scale_x), int(150 * scale_y), int(200 * scale_x), int(30 * scale_y))

        self.combo_tipo = QComboBox(self)
        self.combo_tipo.setGeometry(int(25 * scale_x), int(180 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.combo_tipo.addItems(["MST", "E-ball", "K-NN", "RWC"])

        font = self.combo_tipo.font()
        font.setPointSize(12)
        self.combo_tipo.setFont(font)

    def ajustar_espessura(self, valor):
        """Atualiza a espessura da linha do grafo e redesenha a visualização."""
        self.parent.espessura_grafo = valor
        self.label.setText(f"Espessura da Linha: {valor}")
        self.parent.repaint(self.parent.rect())
        self.parent.update()
        QApplication.processEvents()

    def tentar_gerar_grafo(self):
        tem_minucias = bool(getattr(self.parent, 'minutiae_points', {})) or bool(getattr(self.parent, 'pontos_sobreposta', []))
        if not tem_minucias:
            QMessageBox.warning(self, "Erro", "Não é possível gerar o grafo. Nenhuma minúcia foi detectada ou carregada.")
            return

        tipo = self.combo_tipo.currentIndex()
        self.parent.tipo_grafo = tipo  
        self.parent.gerar_grafo_minucias(tipo)

    def ativar_checkbox_grafo(self):
        self.check_grafo.setEnabled(True)
        self.check_grafo.setChecked(True)

class AtributosDialog(QDialog):
    def __init__(self, parent, scale_x, scale_y):
        super().__init__(parent)
        self.setWindowTitle("Direção das Minúcias")
        self.setGeometry(int(715 * scale_x), int(65 * scale_y), int(250 * scale_x), int(210 * scale_y))

        self.parent_ref = parent

        altura_inicial = getattr(parent, "tamanho_direcao_altura", 70)
        self.label_altura = QLabel(f"Altura da Direção: {altura_inicial}", self)
        self.label_altura.setGeometry(int(15 * scale_x), int(10 * scale_y), int(220 * scale_x), int(20 * scale_y))

        self.slider_altura = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_altura.setGeometry(int(15 * scale_x), int(40 * scale_y), int(200 * scale_x), int(20 * scale_y))
        self.slider_altura.setRange(50, 90)
        self.slider_altura.setValue(altura_inicial)
        self.slider_altura.valueChanged.connect(self.alterar_altura_direcao)

        espessura_inicial = getattr(parent, "tamanho_direcao_espessura", 5)
        self.label_espessura = QLabel(f"Espessura da Direção: {espessura_inicial}", self)
        self.label_espessura.setGeometry(int(15 * scale_x), int(80 * scale_y), int(220 * scale_x), int(20 * scale_y))

        self.slider_espessura = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_espessura.setGeometry(int(15 * scale_x), int(120 * scale_y), int(200 * scale_x), int(20 * scale_y))
        self.slider_espessura.setRange(1, 10)
        self.slider_espessura.setValue(espessura_inicial)
        self.slider_espessura.valueChanged.connect(self.alterar_espessura_direcao)

        self.check_direcao = QCheckBox("Mostrar Direção das Minúcias", self)
        self.check_direcao.setGeometry(int(15 * scale_x), int(160 * scale_y), int(210 * scale_x), int(30 * scale_y))
        self.check_direcao.setChecked(getattr(parent, "mostrar_direcao_minucias", False))
        self.check_direcao.stateChanged.connect(self.toggle_direcao)

    def toggle_direcao(self, state):
        self.parent_ref.mostrar_direcao_minucias = bool(state)
        self.parent_ref.update()

    def alterar_altura_direcao(self, valor):
        self.parent_ref.tamanho_direcao_altura = valor
        self.label_altura.setText(f"Altura da Direção: {valor}")
        self.parent_ref.update()

    def alterar_espessura_direcao(self, valor):
        self.parent_ref.tamanho_direcao_espessura = valor
        self.label_espessura.setText(f"Espessura da Direção: {valor}")
        self.parent_ref.update()

class MatchingDialog(QDialog):
    """Janela de configurações para visualizar e controlar o matching entre minúcias correspondentes."""
    def __init__(self, parent, scale_x, scale_y):
        super().__init__(parent)
        self.setWindowTitle("Matching")
        self.setGeometry(int(15 * scale_x), int(900 * scale_y), int(250 * scale_x), int(170 * scale_y))
        self.parent = parent

        self.label_esp = QLabel(f"Espessura da Linha: {self.parent.espessura_matching}", self)
        self.label_esp.setGeometry(int(25 * scale_x), int(10 * scale_y), int(200 * scale_x), int(30 * scale_y))

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setGeometry(int(25 * scale_x), int(40 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.slider.setRange(1, 10)
        self.slider.setValue(self.parent.espessura_matching)
        self.slider.valueChanged.connect(self.ajustar_espessura)

        self.check_matching = QCheckBox("Exibir Matching", self)
        self.check_matching.setGeometry(int(25 * scale_x), int(75 * scale_y), int(210 * scale_x), int(30 * scale_y))
        self.check_matching.setChecked(False)
        self.check_matching.stateChanged.connect(self.parent.update)

        self.btn_gerar = QPushButton("Gerar Matching", self)
        self.btn_gerar.setGeometry(int(25 * scale_x), int(115 * scale_y), int(200 * scale_x), int(30 * scale_y))
        self.btn_gerar.clicked.connect(self.gerar_matching)

    def ajustar_espessura(self, valor):
        self.parent.espessura_matching = valor
        self.label_esp.setText(f"Espessura da Linha: {valor}")
        self.parent.repaint(self.parent.rect())
        self.parent.update()
        QApplication.processEvents()

    def gerar_matching(self):
        if not hasattr(self.parent, "correspondencias") or not self.parent.correspondencias:
            QMessageBox.warning(self, "Erro", "Nenhuma correspondência encontrada para gerar matching.")
            return

        self.parent.grafo_matching = []

        for idx, (x2, y2) in self.parent.correspondencias.items():
            if idx < 1000:
                if idx in self.parent.minutiae_points:
                    x1, y1, _ = self.parent.minutiae_points[idx]
                    self.parent.grafo_matching.append(((x1, y1), (x2, y2)))
                else:
                    offset = len(self.parent.minutiae_points)
                    i = idx - offset - 1
                    if 0 <= i < len(self.parent.pontos_minucia_manual):
                        x1, y1 = self.parent.pontos_minucia_manual[i]
                        self.parent.grafo_matching.append(((x1, y1), (x2, y2)))

            elif 1000 <= idx < 2000:
                i = idx - 1000 - 1
                if 0 <= i < len(self.parent.pontos_delta):
                    x1, y1 = self.parent.pontos_delta[i]
                    self.parent.grafo_matching.append(((x1, y1), (x2, y2)))

            elif 2000 <= idx < 3000:
                i = idx - 2000 - 1
                if 0 <= i < len(self.parent.pontos_core):
                    x1, y1 = self.parent.pontos_core[i]
                    self.parent.grafo_matching.append(((x1, y1), (x2, y2)))

            else:
                offset = max(self.parent.minutiae_points.keys(), default=0)
                i = idx - offset
                if 0 <= i < len(self.parent.pontos_minucia_manual):
                    x1, y1 = self.parent.pontos_minucia_manual[i]
                    self.parent.grafo_matching.append(((x1, y1), (x2, y2)))

        self.check_matching.setChecked(True)
        self.parent.atualizar_status_lateral()
        self.parent.update()

app = QApplication(sys.argv)  
screen_geometry = QGuiApplication.primaryScreen().geometry()
screen_width = screen_geometry.width()
screen_height = screen_geometry.height()
scale_x = screen_width / 1920
scale_y = screen_height / 1080

stylesheet = f"""
    QWidget {{
        background-color: #2D2D2D;
        color: #FFFFFF;
    }}
    QPushButton {{
        background-color: #3A3A3A;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 5px;
        font-size: {int(13 * scale_x)}px; 
    }}
    QPushButton:hover {{
        background-color: #555; 
    }}
    QPushButton:pressed {{
        background-color: #777;  
        border: 1px solid #888;  
    }}
    QLineEdit {{
        background-color: #3A3A3A;
        color: #FFFFFF;
        border: 1px solid #555;
        padding: 2px;
        font-size: {int(13 * scale_x)}px; 
    }}
    QLabel {{
        color: #FFFFFF;
        font-size: {int(13 * scale_x)}px; 
    }}
    
    QCheckBox {{
        spacing: 5px; 
        font-size: {int(13 * scale_x)}px; 
    }}

    QCheckBox::indicator {{
        width: {int(12 * scale_x)}px; 
        height: {int(12 * scale_x)}px;
        border-radius: {int(4 * scale_x)}px; 
        border: 2px solid #555;
        background-color: #2D2D2D; 
    }}

    QCheckBox::indicator:hover {{
        border: 2px solid #888;
        background-color: #555; 
    }}

    QCheckBox::indicator:checked {{
        background-color: #0078D7;
        border: 2px solid #0078D7; 
    }}

    QCheckBox::indicator:unchecked {{
        background-color: #2D2D2D; 
        border: 2px solid #555;
    }}
"""

app.setStyle("Fusion")
app.setStyleSheet(stylesheet)

janela = Layout()
janela.showFullScreen()
sys.exit(app.exec())
