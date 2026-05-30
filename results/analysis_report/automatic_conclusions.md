# Wnioski automatycznie wygenerowane na podstawie wyników

## Parametry eksperymentu
We wszystkich eksperymentach wykorzystano spektrogramy melowe o liczbie pasm n_mels=128, długości próbek 2.0 s oraz częstotliwości próbkowania 16000 Hz.
Dla eksperymentów typu binary zastosowano: noise_std=0.001, kalman_q=1e-05, kalman_r=1e-06.
Dla eksperymentów typu multiclass zastosowano: noise_std=0.0005, kalman_q=1e-06, kalman_r=1e-05.

## Wyniki testowe z checkpointów
- binary / original: accuracy=98.80%, precision_macro=0.9558, recall_macro=0.9865, F1_macro=0.9704, loss=0.0360.
- binary / noisy: accuracy=99.49%, precision_macro=0.9837, recall_macro=0.9903, F1_macro=0.9870, loss=0.0266.
- binary / noisy_kalman: accuracy=98.63%, precision_macro=0.9576, recall_macro=0.9742, F1_macro=0.9657, loss=0.0800.
- multiclass / original: accuracy=88.62%, precision_macro=0.7526, recall_macro=0.9343, F1_macro=0.7966, loss=0.3374.
- multiclass / noisy: accuracy=98.41%, precision_macro=0.9561, recall_macro=0.9361, F1_macro=0.9457, loss=0.0879.
- multiclass / noisy_kalman: accuracy=98.63%, precision_macro=0.9552, recall_macro=0.9526, F1_macro=0.9535, loss=0.0557.

## Przebieg uczenia
- binary_noisy: najlepsza wartość val_acc=0.9954 w epoce 15; najniższy val_loss=0.0282.
- binary_noisy_kalman: najlepsza wartość val_acc=0.9846 w epoce 5; najniższy val_loss=0.0561.
- binary_original: najlepsza wartość val_acc=0.9897 w epoce 18; najniższy val_loss=0.0442.
- multiclass_noisy: najlepsza wartość val_acc=0.9789 w epoce 5; najniższy val_loss=0.2124.
- multiclass_noisy_kalman: najlepsza wartość val_acc=0.9886 w epoce 12; najniższy val_loss=0.0448.
- multiclass_original: najlepsza wartość val_acc=0.9761 w epoce 6; najniższy val_loss=0.0898.

## Wpływ szumu i filtru Kalmana
- binary / noisy: względem wariantu original zmiana accuracy wyniosła 0.68 p.p., a zmiana F1_macro wyniosła 0.0165. Parametry dla tego typu danych: noise_std=0.001, kalman_q=1e-05, kalman_r=1e-06.
- binary / noisy_kalman: względem wariantu original zmiana accuracy wyniosła -0.17 p.p., a zmiana F1_macro wyniosła -0.0047. Parametry dla tego typu danych: noise_std=0.001, kalman_q=1e-05, kalman_r=1e-06.
- multiclass / noisy: względem wariantu original zmiana accuracy wyniosła 9.79 p.p., a zmiana F1_macro wyniosła 0.1491. Parametry dla tego typu danych: noise_std=0.0005, kalman_q=1e-06, kalman_r=1e-05.
- multiclass / noisy_kalman: względem wariantu original zmiana accuracy wyniosła 10.02 p.p., a zmiana F1_macro wyniosła 0.1569. Parametry dla tego typu danych: noise_std=0.0005, kalman_q=1e-06, kalman_r=1e-05.
