import numpy as np
from PIL import ImageGrab
import cv2
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController

# Tastensimulator und Maussimulator initialisieren
keyboard_controller = Controller()
mouse_controller = MouseController()
script_running = True  # Globale Variable zum Beenden des Skripts

# Definiere den Bereich des Screens, der die Minikarte unten links umfasst
bbox = (0, 860, 320, 1080)  # Angepasster Bereich basierend auf deinem Screenshot

# Definiere die RGB-Werte für Gelb und eine Toleranz
yellow_rgb = [232, 240, 39]  # RGB-Wert des gelben Punktes, basierend auf deinem Screenshot
tolerance = 40  # Toleranz für die RGB-Werte

# Berechne die unteren und oberen Grenzwerte für Gelb im RGB-Farbraum
lower_yellow = np.array([max(0, c - tolerance) for c in yellow_rgb], dtype=np.uint8)
upper_yellow = np.array([min(255, c + tolerance) for c in yellow_rgb], dtype=np.uint8)

# Mindestkonturfläche für den gelben Punkt
min_contour_area = 50

# Bewegungspuffer und Schwellenwert für das Anhalten
target_reached_threshold = 2  # Schwellenwert für exakte Positionierung
close_distance_threshold = 20  # Schwellenwert für präzise Schritte
far_distance_threshold = 100  # Ab dieser Distanz wird eine längere Bewegung ausgeführt
movement_duration_far = 0.5  # Längere Bewegungsdauer für große Abstände
movement_duration_close = 0.1  # Kürzere Bewegung für Präzision in der Nähe

# Blockade-Überwachungsparameter
stuck_duration_threshold = 3  # Sekunden, nach denen festgefahren erkannt wird
stuck_start_time = None       # Zeitpunkt der letzten Distanzänderung
last_distance = None          # Letzte gemessene Distanz zum Ziel

# Funktion, um eine Richtungstaste zusammen mit der Sprint-Taste für eine bestimmte Dauer gedrückt zu halten
def hold_key_with_sprint(key, duration):
    keyboard_controller.press(Key.shift)
    keyboard_controller.press(key)
    time.sleep(duration)
    keyboard_controller.release(key)
    keyboard_controller.release(Key.shift)

# Funktion, um eine Richtungstaste ohne die Sprinttaste für eine bestimmte Dauer gedrückt zu halten
def hold_key_without_sprint(key, duration):
    keyboard_controller.press(key)
    time.sleep(duration)
    keyboard_controller.release(key)

# Funktion für das Rückwärts- und Seitwärtsbewegen bei Festhängen und Kamera-Drehung
def move_backwards_and_side_with_rotation():
    # Rückwärts- und Seitwärtsbewegung
    hold_key_without_sprint('s', 0.3)
    hold_key_without_sprint('d', 0.3)

    # Drehe die Kamera um 180 Grad mit mehreren kleinen Bewegungen
    total_rotation_steps = 40  # Anzahl der kleinen Schritte für 180-Grad-Drehung
    for _ in range(total_rotation_steps):
        mouse_controller.move(15, 0)  # Bewegt die Maus leicht nach rechts
        time.sleep(0.01)  # Kleine Pause für gleichmäßige Bewegung

# Funktion zum Beenden des Skripts bei Drücken von ESC
def on_press(key):
    global script_running
    if key == keyboard.Key.esc:
        script_running = False
        return False  # Stoppe den Listener

# Startet den Keyboard-Listener im Hintergrund, damit ESC das Skript jederzeit beendet
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Initialisiere last_time vor der Schleife
last_time = time.time()

while script_running:
    # Screenshot des gesamten Bildschirms
    full_screen = np.array(ImageGrab.grab())
    full_screen_rgb = cv2.cvtColor(full_screen, cv2.COLOR_BGR2RGB)
    screen = full_screen[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # Maske für den gelben Punkt im RGB-Farbraum
    yellow_mask = cv2.inRange(screen, lower_yellow, upper_yellow)
    yellow_mask = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
    cv2.imshow("Gelber Punkt Maske", yellow_mask)

    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = [cnt for cnt in yellow_contours if cv2.contourArea(cnt) > min_contour_area]

    if yellow_contours:
        # Berechne den exakten Schwerpunkt (Centroid) des größten gelben Punktes
        largest_yellow_contour = max(yellow_contours, key=cv2.contourArea)
        M = cv2.moments(largest_yellow_contour)
        if M["m00"] != 0:
            target_x = int(M["m10"] / M["m00"])
            target_y = int(M["m01"] / M["m00"])
        else:
            target_x, target_y = 0, 0  # Fallback bei Fehler

        print(f"Gelber Punkt Schwerpunkt bei: ({target_x}, {target_y})")
        
        player_x = bbox[2] // 2
        player_y = bbox[3] // 2
        delta_x = target_x - player_x
        delta_y = target_y - player_y
        current_distance = np.sqrt(delta_x**2 + delta_y**2)

        # Blockadeerkennung: Überprüfe, ob die Distanz unverändert bleibt
        if last_distance is not None and abs(current_distance - last_distance) < 1:
            # Wenn stuck_start_time noch nicht gesetzt ist, setze sie jetzt
            if stuck_start_time is None:
                stuck_start_time = time.time()
            elif time.time() - stuck_start_time > stuck_duration_threshold:
                print("Blockade erkannt! Bewege rückwärts und zur Seite mit 180-Grad-Drehung.")
                move_backwards_and_side_with_rotation()
                stuck_start_time = None  # Reset der Blockadezeit
        else:
            stuck_start_time = None  # Setzt stuck_start_time zurück, wenn sich die Distanz ändert

        last_distance = current_distance  # Update last_distance

        # Bewegungsdauer basierend auf der Distanz anpassen
        if current_distance > far_distance_threshold:
            duration = movement_duration_far
            use_sprint = True
        elif current_distance > close_distance_threshold:
            duration = movement_duration_far / 2
            use_sprint = True
        else:
            duration = movement_duration_close
            use_sprint = False

        # Kontinuierliche Bewegung nach links oder rechts
        if abs(delta_x) > target_reached_threshold:
            if delta_x < 0:
                if use_sprint:
                    hold_key_with_sprint('a', duration)
                else:
                    hold_key_without_sprint('a', duration)
            elif delta_x > 0:
                if use_sprint:
                    hold_key_with_sprint('d', duration)
                else:
                    hold_key_without_sprint('d', duration)

        # Kontinuierliche Bewegung nach oben oder unten
        if abs(delta_y) > target_reached_threshold:
            if delta_y < 0:
                if use_sprint:
                    hold_key_with_sprint('w', duration)
                else:
                    hold_key_without_sprint('w', duration)
            elif delta_y > 0:
                if use_sprint:
                    hold_key_with_sprint('s', duration)
                else:
                    hold_key_without_sprint('s', duration)

    else:
        print("Kein gelber Punkt gefunden.")

    cv2.rectangle(full_screen_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imshow("Bildschirm mit markiertem Minimap-Bereich", full_screen_rgb)

    print(f'Loop took {time.time() - last_time} seconds')
    last_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # Kürzere Pause für flüssigere Bewegung

cv2.destroyAllWindows()
listener.stop()  # Beende den Listener, wenn die Schleife abgebrochen wird
