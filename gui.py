import pygame
import random
import csv
from main import predict

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Prédiction Maladie Cardiaque")

FONT = pygame.font.SysFont('Arial', 22)
BIG_FONT = pygame.font.SysFont('Arial', 28)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
BLACK = (0, 0, 0)
BLUE = (0, 120, 215)
RED = (200, 0, 0)

DROPDOWN_OPTION_HEIGHT = 30

fields = [
    {"key": "Age", "label": "Âge", "type": "number"},
    {"key": "Sex", "label": "Sexe", "type": "dropdown", "options": ["M", "F"]},
    {"key": "ChestPainType", "label": "Douleur thoracique", "type": "dropdown",
     "options": ["Angine typique", "Angine atypique", "Douleur non angineuse", "Asymptomatique"]},
    {"key": "RestingBP", "label": "Tension au repos", "type": "number"},
    {"key": "Cholesterol", "label": "Cholestérol", "type": "number"},
    {"key": "FastingBS", "label": "Glycémie à jeun", "type": "number"},
    {"key": "RestingECG", "label": "Électrocardiogramme", "type": "dropdown",
     "options": ["Normal", "Anomalies ST-T", "Hypertrophie VG"]},
    {"key": "MaxHR", "label": "Fréquence cardiaque max", "type": "number"},
    {"key": "ExerciseAngina", "label": "Angine d'effort", "type": "dropdown", "options": ["Oui", "Non"]},
    {"key": "Oldpeak", "label": "Oldpeak", "type": "number"},
    {"key": "ST_Slope", "label": "Pente ST", "type": "dropdown", "options": ["Ascendante", "Plate", "Descendante"]},
    {"key": "model", "label": "Sélectionnez le model à utiliser", "type": "dropdown", "options": ["KNN", "XGBoost", "Neural Network"]}
]

inputs = {field["key"]: "" for field in fields}
dropdown_open = None
active_field = None
result_text = ""

# Transforme les entrées utilisateur dans le même format que les données initiales
def transform_input(inputs):
    key_map = {
        'ChestPainType': 'chestpaintype',
        'RestingECG': 'restingecg',
        'ExerciseAngina': 'exerciseangina',
        'ST_Slope': 'st_slop'
    }

    chest_pain_map = {
        "Angine typique": "TA",
        "Angine atypique": "ATA",
        "Douleur non angineuse": "NAP",
        "Asymptomatique": "ASY"
    }

    resting_ecg_map = {
        "Normal": "Normal",
        "Anomalies ST-T": "ST",
        "Hypertrophie VG": "LVH"
    }

    exercise_angina_map = {
        "Oui": "Y",
        "Non": "N"
    }

    st_slope_map = {
        "Ascendante": "Up",
        "Plate": "Flat",
        "Descendante": "Down"
    }

    transformed = inputs.copy()

    for key, internal_key in key_map.items():
        if key in transformed:
            value = transformed[key]
            if internal_key == 'chestpaintype':
                transformed[key] = chest_pain_map.get(value, value)
            elif internal_key == 'restingecg':
                transformed[key] = resting_ecg_map.get(value, value)
            elif internal_key == 'exerciseangina':
                transformed[key] = exercise_angina_map.get(value, value)
            elif internal_key == 'st_slop':
                transformed[key] = st_slope_map.get(value, value)

    # On enlève la clé 'model' car elle n'est pas nécessaire pour la prédiction
    transformed.pop('model', None)
    return transformed

# Converti le dictionnaire "data" en CSV tout simplement
def dict_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

def draw_text(surface, text, pos, font, color=BLACK):
    surface.blit(font.render(text, True, color), pos)

def predict_dummy_model(data):
    return random.choice([0, 1]), random.choice([0, 1])

def main():
    global active_field, dropdown_open, result_text
    scroll_offset = 0
    scroll_speed = 30
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(WHITE)
        y_offset = 30 + scroll_offset

        for field in fields:
            key, label, ftype = field["key"], field["label"], field["type"]
            draw_text(screen, f"{label} :", (50, y_offset), FONT)

            rect = pygame.Rect(400, y_offset, 300, 30)
            if ftype == "dropdown":
                pygame.draw.rect(screen, LIGHT_GRAY if dropdown_open == key else GRAY, rect)
                draw_text(screen, inputs[key] if inputs[key] else "Sélectionner...", (410, y_offset + 5), FONT, BLACK if inputs[key] else (120, 120, 120))

                if dropdown_open == key:
                    for i, option in enumerate(field["options"]):
                        opt_rect = pygame.Rect(400, y_offset + DROPDOWN_OPTION_HEIGHT * (i + 1), 300, DROPDOWN_OPTION_HEIGHT)
                        pygame.draw.rect(screen, WHITE, opt_rect)
                        pygame.draw.rect(screen, GRAY, opt_rect, 1)
                        draw_text(screen, option, (410, y_offset + 5 + DROPDOWN_OPTION_HEIGHT * (i + 1)), FONT)

            else:
                pygame.draw.rect(screen, BLUE if active_field == key else GRAY, rect, 2)
                draw_text(screen, inputs[key], (410, y_offset + 5), FONT)

            y_offset += 50
            if dropdown_open == key:
                y_offset += DROPDOWN_OPTION_HEIGHT * len(field["options"])

        predict_rect = pygame.Rect(50, y_offset + 20, 200, 40)
        pygame.draw.rect(screen, BLUE, predict_rect)
        draw_text(screen, "Prédire", (110, y_offset + 30), FONT, WHITE)

        draw_text(screen, result_text, (300, y_offset + 30), BIG_FONT, RED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset += event.y * scroll_speed

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(transform_input(inputs))
                elif active_field:
                    if event.key == pygame.K_BACKSPACE:
                        inputs[active_field] = inputs[active_field][:-1]
                    elif event.key == pygame.K_RETURN:
                        active_field = None
                    else:
                        inputs[active_field] += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                y_cursor = 30 + scroll_offset
                clicked = False

                for field in fields:
                    key, ftype = field["key"], field["type"]
                    rect = pygame.Rect(400, y_cursor, 300, 30)

                    if rect.collidepoint(mx, my):
                        if ftype == "dropdown":
                            dropdown_open = key if dropdown_open != key else None
                        else:
                            active_field, dropdown_open = key, None
                        clicked = True
                        break

                    if dropdown_open == key and ftype == "dropdown":
                        for i, option in enumerate(field["options"]):
                            opt_rect = pygame.Rect(400, y_cursor + DROPDOWN_OPTION_HEIGHT * (i + 1), 300, DROPDOWN_OPTION_HEIGHT)
                            if opt_rect.collidepoint(mx, my):
                                inputs[key] = option
                                dropdown_open = None
                                clicked = True
                                break
                        y_cursor += DROPDOWN_OPTION_HEIGHT * len(field["options"])
                    y_cursor += 50

                if not clicked:
                    active_field = None

                # Lorsque l'utilisateur clique sur le bouton de prédiction
                if predict_rect.collidepoint(mx, my):
                    if all(inputs[f["key"]] for f in fields):
                        dict_to_csv(transform_input(inputs), 'newPatient.csv') # Enregistre les données dans un fichier CSV
                        model = inputs["model"] # Récupère le modèle sélectionné
                        result = None

                        # Appelle la fonction de prédiction en fonction du modèle sélectionné
                        if model == "KNN":
                            result = predict("knn")
                        elif model == "XGBoost":
                            result = "lmao"
                        elif model == "Neural Network":
                            result = predict("neural")
                        result_text = ""
                        if model == "KNN":
                            result_text += "KNN: "
                        elif model == "XGBoost":
                            result_text += "XGBoost Optimisé: "
                        elif model == "Neural Network":
                            result_text += "Neural Network: "

                        # check resultat
                        if result == 1:
                            result_text += "Malade"
                        elif result == "lmao":
                            result_text += "Pas implémenté"
                        else:
                            result_text += "Sain"
                        # knn_pred, xgb_pred = predict_dummy_model(inputs)
                        # result_text = f"KNN: {'Malade' if knn_pred else 'Sain'} | XGBoost: {'Malade' if xgb_pred else 'Sain'}"
                    else:
                        result_text = "Veuillez remplir tous les champs."

        dropdown_extra_height = 0
        if dropdown_open:
            for field in fields:
                if field["key"] == dropdown_open and field["type"] == "dropdown":
                    dropdown_extra_height += DROPDOWN_OPTION_HEIGHT * len(field["options"])
                    break

        total_form_height = y_offset + dropdown_extra_height + 200

        max_scroll = max(0, total_form_height - HEIGHT)
        scroll_offset = min(max(scroll_offset, -max_scroll), 0)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()