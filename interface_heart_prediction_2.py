
import pygame
import random

# Initialisation de Pygame
pygame.init()
pygame.font.init()

# Dimensions de la fenêtre
WIDTH, HEIGHT = 900, 850
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Prédiction Maladie Cardiaque")

FONT = pygame.font.SysFont('Arial', 22)
BIG_FONT = pygame.font.SysFont('Arial', 28)

# Couleurs
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 120, 215)
RED = (200, 0, 0)

# Champs requis (avec noms + descriptions formatés)
fields = {
    "age": "Âge",
    "sex": "Sexe (M/F)",
    "chestpaintype": "Douleur thoracique (TA, ATA, NAP, ASY)",
    "restingbp": "Tension au repos (mm Hg)",
    "cholesterol": "Cholestérol (mg/dl)",
    "fastingbs": "Glycémie à jeun (1 >120 mg/dl, sinon 0)",
    "restingecg": "ECG au repos (Normal, ST, LVH)",
    "maxhr": "Fréquence cardiaque max",
    "exerciseangina": "Angine d'effort (Y/N)",
    "oldpeak": "Oldpeak (décroissance ST)",
    "st_slop": "Pente ST (Up, Flat, Down)"
}

inputs = {key: '' for key in fields}
active_field = None
result_text = ""

def draw_text(surface, text, pos, font, color=BLACK):
    label = font.render(text, True, color)
    surface.blit(label, pos)

def predict_dummy_model(data):
    prediction_knn = random.choice([0, 1])
    prediction_xgb = random.choice([0, 1])
    return prediction_knn, prediction_xgb

def main():
    global active_field, result_text

    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(WHITE)

        # Afficher les champs
        y_offset = 30
        for key, label in fields.items():
            draw_text(screen, f"{label} :", (50, y_offset), FONT)
            pygame.draw.rect(screen, BLUE if active_field == key else GRAY, (400, y_offset, 250, 30), 2)
            draw_text(screen, inputs[key], (410, y_offset + 5), FONT)
            y_offset += 50

        # Bouton prédiction
        pygame.draw.rect(screen, BLUE, (50, y_offset + 20, 200, 40))
        draw_text(screen, "Prédire", (110, y_offset + 30), FONT, WHITE)

        # Résultat
        draw_text(screen, result_text, (300, y_offset + 30), BIG_FONT, RED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                y_check = 30
                active_field = None
                for key in fields:
                    if 400 <= mx <= 650 and y_check <= my <= y_check + 30:
                        active_field = key
                        break
                    y_check += 50
                if 50 <= mx <= 250 and y_check + 20 <= my <= y_check + 60:
                    if all(inputs[key] for key in fields):
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
