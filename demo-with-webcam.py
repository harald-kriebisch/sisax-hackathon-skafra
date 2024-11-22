import aisisax.object_detection.lsa_interface as aisax_object_detection
import aisisax.llm.openai_connector as aisax_openai
import aisisax.llm.ollama_connector as aisax_ollama
import cv2

from IPython.display import display
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # Save the image to assets
        img_name = "assets/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        # See if a skat card was found and show the resulting image detection
        res = aisax_object_detection.call_lsa(img_name, "card")
        display(res)

        game_dic = {
            "initial_hand_cards": []
        }

        # Let chatgpt read the saved image and let it describe it
        openai_result = aisax_openai.generate_multimodal_answer("""
        Welche Skatkarten kannst du erkennen? returne ein array of objects wobei jedes objekt eine karte repräsentiert. 
        jedes objekt hat zwei attribute "symbol" and "kind".

        "kind" können ausschließlich items aus der folgenden liste sein: ["sieben", "acht", "neun", "zehn", "bube", "Dame", "könig", "ass"];
        "symbol" können ausschließlich items aus der folgenden liste sein: ["kreuz", "blatt", "herz", "karo"]

        Der Output sollte sein, e.g. 
        [
        {
            symbol: "herz",
            kind: "sieben"
        }
        ]
        """, image_path=img_name)
        print(openai_result)

        img_counter += 1

cam.release()
cv2.destroyAllWindows()