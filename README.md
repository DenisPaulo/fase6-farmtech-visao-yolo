# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP" border="0" width="40%" height="40%"></a>
</p>

<br>

# 📌 FarmTech Solutions — Fase 6: Colheita de Soluções Inteligentes (Visão Computacional com YOLOv8)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

## 👨‍🎓 Integrantes
- [Jeliel Cardoso](https://www.linkedin.com/in/jelielcardoso/) — **RM572665**
- [Denis Paulo Dias da Silva](https://www.linkedin.com/in/denispaulodiassilva/) — **RM573278**
- [Deweyne Reuel](https://www.linkedin.com/in/deweyne-reuel-0695522a8/) — **RM573755**

## 👩‍🏫 Professores
### Tutor(a)
- [Sabrina Otoni](https://www.linkedin.com/in/sabrina-otoni-22525519b/)
### Coordenador(a)
- [André Godoi](https://www.linkedin.com/in/andregodoichiovato/)

---

## 📖 Descrição

Na **Fase 6 (Colheita de Soluções Inteligentes)** implementamos o subsistema de **Visão Computacional com YOLOv8** para inspeção fitossanitária automatizada dos talhões de café da **FarmTech Solutions**.

O detector analisa imagens das folhas e frutos coletadas em campo e identifica 5 classes:
- `folha_saudavel` (Operação normal)
- `fruto_maduro` (Planejamento de colheita seletiva)
- `deficit_hidrico` (Severidade MÉDIA — cruzar com sensores de umidade do solo)
- `ferrugem_cafeeiro` (*Hemileia vastatrix*, Severidade ALTA — controle fungicida imediato)
- `praga_broca` (*Hypothenemus hampei*, Severidade ALTA — manejo integrado de pragas)

As detecções são gravadas em `reports/deteccoes_recentes.json`, alimentando diretamente a **Dashboard Integrada** e o **Serviço de Mensageria AWS SNS** na Fase 7.

<p align="center">
  <img src="assets/deteccoes_yolo_fase6.png" alt="Detecções YOLOv8 nos talhões de café" width="90%"/>
</p>

---

## ⚙️ Como Executar

```bash
pip install -r requirements.txt
python -m src.detector_lavoura --demo
pytest -q
```

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
