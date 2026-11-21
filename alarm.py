
import winsound

print("🚨 ALARM ACTIVATED!")

winsound.PlaySound(
    "alarm.wav",
    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
)
