"""
Real-Time Guitar Tuner
Standard tuning: E2 A2 D3 G3 B3 E4

Listens to the microphone, estimates the fundamental frequency,
finds the nearest guitar string, and shows whether it is
flat, sharp, or in tune.
"""

import math
from collections import deque

import numpy as np
import sounddevice as sd




SAMPLE_RATE = 44100
BLOCK_SIZE = 8192

MIN_FREQ = 70.0
MAX_FREQ = 400.0

VOLUME_THRESHOLD = 0.008

TUNING_TOLERANCE = 5.0

SMOOTHING_SIZE = 5


# Standard guitar tuning
GUITAR_STRINGS = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63,
}


# Store recent frequencies
frequency_history = deque(maxlen=SMOOTHING_SIZE)




def frequency_to_cents(frequency, target_frequency):
    """
    Calculate the pitch difference in cents.

    0 cents = perfectly tuned
    negative = flat
    positive = sharp
    """

    return 1200 * math.log2(frequency / target_frequency)


def find_nearest_string(frequency):
    """Find the guitar string closest to the detected pitch."""

    closest_note = None
    closest_frequency = None
    smallest_difference = float("inf")

    for note, target_frequency in GUITAR_STRINGS.items():

        cents = abs(
            frequency_to_cents(
                frequency,
                target_frequency
            )
        )

        if cents < smallest_difference:
            smallest_difference = cents
            closest_note = note
            closest_frequency = target_frequency

    cents = frequency_to_cents(
        frequency,
        closest_frequency
    )

    return closest_note, closest_frequency, cents




def detect_pitch(samples):
    """
    Estimate the fundamental frequency using autocorrelation.
    """

    # Convert to one-dimensional array
    samples = np.asarray(samples).flatten()

    if len(samples) == 0:
        return None

    
    rms = np.sqrt(np.mean(samples ** 2))

    if rms < VOLUME_THRESHOLD:
        return None

   
    samples = samples - np.mean(samples)

   
    window = np.hanning(len(samples))
    windowed = samples * window

    
    correlation = np.correlate(
        windowed,
        windowed,
        mode="full"
    )

    
    correlation = correlation[len(correlation) // 2:]

    
    min_lag = int(SAMPLE_RATE / MAX_FREQ)
    max_lag = int(SAMPLE_RATE / MIN_FREQ)

    search_region = correlation[min_lag:max_lag]

    if len(search_region) == 0:
        return None

    peak_index = np.argmax(search_region)

    peak_lag = peak_index + min_lag

    
    if 1 <= peak_lag < len(correlation) - 1:

        y1 = correlation[peak_lag - 1]
        y2 = correlation[peak_lag]
        y3 = correlation[peak_lag + 1]

        denominator = y1 - 2 * y2 + y3

        if denominator != 0:

            adjustment = 0.5 * (y1 - y3) / denominator
            peak_lag = peak_lag + adjustment

    

    frequency = SAMPLE_RATE / peak_lag

    if not MIN_FREQ <= frequency <= MAX_FREQ:
        return None

    return float(frequency)




def smooth_frequency(frequency):
    """
    Smooth several recent measurements to reduce jumping.
    """

    frequency_history.append(frequency)

    return float(np.median(frequency_history))




def create_tuning_bar(cents):
    """
    Create a visual tuning indicator.

    Example:

    FLAT  -----------|--●--|----------- SHARP
                         ^
                       IN TUNE
    """

    width = 41

    # Limit display to +/- 50 cents
    display_cents = max(-50, min(50, cents))

    position = int(
        ((display_cents + 50) / 100) * (width - 1)
    )

    center = width // 2

    bar = ["-"] * width

    # Center represents perfect tuning
    bar[center] = "|"

    # Detected position
    bar[position] = "●"

    return "".join(bar)


def tuning_instruction(cents):

    if abs(cents) <= TUNING_TOLERANCE:
        return "✓ IN TUNE"

    elif cents < 0:
        return "↑ TUNE UP"

    else:
        return "↓ TUNE DOWN"


def display_result(note, frequency, target, cents):

    bar = create_tuning_bar(cents)

    print("\033[2J\033[H", end="")

    print("=" * 60)
    print("                    GUITAR TUNER")
    print("=" * 60)

    print()
    print(f"                         {note}")
    print()
    print(f"                   {frequency:7.2f} Hz")
    print()
    print(f"               Target: {target:.2f} Hz")
    print()

    print(" FLAT                                        SHARP")
    print(f" {bar}")

    print()
    print(f"                    {cents:+.1f} cents")
    print()
    print(f"                    {tuning_instruction(cents)}")

    print()
    print("=" * 60)
    print("             Ctrl+C to stop the tuner")



def audio_callback(indata, frames, time, status):

    if status:
        print(status)

    samples = indata[:, 0]

    frequency = detect_pitch(samples)

    if frequency is None:
        return

    # Smooth frequency readings
    frequency = smooth_frequency(frequency)

    note, target, cents = find_nearest_string(frequency)

    display_result(
        note,
        frequency,
        target,
        cents
    )



def main():

    print("=" * 60)
    print("              REAL-TIME GUITAR TUNER")
    print("=" * 60)

    print()
    print("Standard guitar tuning:")
    print()
    print("        E2   A2   D3   G3   B3   E4")
    print()
    print("Play ONE open string at a time.")
    print("Let the string ring clearly.")
    print()
    print("Starting microphone...")
    print()

    try:

        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=audio_callback
        ):

            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:

        print("\nTuner stopped.")

    except Exception as error:

        print("\nCould not start the tuner.")
        print(error)

        print(
            "\nCheck that Cursor/Terminal has permission "
            "to access your microphone."
        )




if __name__ == "__main__":
    main()