import pygame
import random

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 900, 950
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

# Champs avec type et options
fields = [
    {"key": "sex", "label": "Sexe", "type": "dropdown", "options": ["M", "F"]},
    {"key": "chestpaintype", "label": "Douleur thoracique", "type": "dropdown",
     "options": ["Angine typique", "Angine atypique", "Douleur non angineuse", "Asymptomatique"]},
    {"key": "restingbp", "label": "Tension au repos", "type": "number"},
    {"key": "cholesterol", "label": "Cholestérol", "type": "number"},
    {"key": "fastingbs", "label": "Glycémie à jeun", "type": "number"},
    {"key": "restingecg", "label": "Électrocardiogramme", "type": "dropdown",
     "options": ["Normal", "Anomalies ST-T", "Hypertrophie VG"]},
    {"key": "maxhr", "label": "Fréquence cardiaque max", "type": "number"},
    {"key": "exerciseangina", "label": "Angine d'effort", "type": "dropdown", "options": ["Oui", "Non"]},
    {"key": "oldpeak", "label": "Oldpeak", "type": "number"},
    {"key": "st_slop", "label": "Pente ST", "type": "dropdown", "options": ["Ascendante", "Plate", "Descendante"]}
]

inputs = {field["key"]: "" for field in fields}
dropdown_open = None
active_field = None
result_text = ""

def draw_text(surface, text, pos, font, color=BLACK):
    surface.blit(font.render(text, True, color), pos)

def predict_dummy_model(data):
    return random.choice([0, 1]), random.choice([0, 1])

def main():
    global active_field, dropdown_open, result_text
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(WHITE)
        y_offset = 30

        for field in fields:
            key, label, ftype = field["key"], field["label"], field["type"]
            draw_text(screen, f"{label} :", (50, y_offset), FONT)

            rect = pygame.Rect(400, y_offset, 300, 30)
            if ftype == "dropdown":
                pygame.draw.rect(screen, LIGHT_GRAY if dropdown_open == key else GRAY, rect)
                draw_text(screen, inputs[key] if inputs[key] else "Sélectionner...", (410, y_offset + 5), FONT, BLACK if inputs[key] else (120,120,120))

                if dropdown_open == key:
                    for i, option in enumerate(field["options"]):
                        opt_rect = pygame.Rect(400, y_offset + 30 * (i + 1), 300, 30)
                        pygame.draw.rect(screen, WHITE, opt_rect)
                        pygame.draw.rect(screen, GRAY, opt_rect, 1)
                        draw_text(screen, option, (410, y_offset + 5 + 30 * (i + 1)), FONT)

            else:
                pygame.draw.rect(screen, BLUE if active_field == key else GRAY, rect, 2)
                draw_text(screen, inputs[key], (410, y_offset + 5), FONT)

            y_offset += 50
            if dropdown_open == key:
                y_offset += 30 * len(field["options"])

        # Bouton de prédiction
        predict_rect = pygame.Rect(50, y_offset + 20, 200, 40)
        pygame.draw.rect(screen, BLUE, predict_rect)
        draw_text(screen, "Prédire", (110, y_offset + 30), FONT, WHITE)

        draw_text(screen, result_text, (300, y_offset + 30), BIG_FONT, RED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(inputs)  # Print inputs when spacebar is pressed
                elif active_field:
                    if event.key == pygame.K_BACKSPACE:
                        inputs[active_field] = inputs[active_field][:-1]
                    elif event.key == pygame.K_RETURN:
                        active_field = None
                    else:
                        inputs[active_field] += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                y_cursor = 30
                clicked = False
                # dropdown_open = None if dropdown_open else dropdown_open
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
                            opt_rect = pygame.Rect(400, y_cursor + 30 * (i + 1), 300, 30)
                            if opt_rect.collidepoint(mx, my):
                                inputs[key] = option
                                dropdown_open = None
                                clicked = True
                                break
                        y_cursor += 30 * len(field["options"])
                    y_cursor += 50

                if not clicked:
                    active_field = None

                if predict_rect.collidepoint(mx, my):
                    if all(inputs[f["key"]] for f in fields):
                        knn_pred, xgb_pred = predict_dummy_model(inputs)
                        result_text = f"KNN: {'Malade' if knn_pred else 'Sain'} | XGBoost: {'Malade' if xgb_pred else 'Sain'}"
                    else:
                        result_text = "Veuillez remplir tous les champs."

            elif event.type == pygame.KEYDOWN and active_field:
                if event.key == pygame.K_BACKSPACE:
                    inputs[active_field] = inputs[active_field][:-1]
                elif event.key == pygame.K_RETURN:
                    active_field = None
                else:
                    inputs[active_field] += event.unicode

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
