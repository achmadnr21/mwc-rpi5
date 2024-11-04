import gpiod
import time

IRLED= 16
chip = gpiod.Chip('gpiochip4')

led_line = chip.get_line(IRLED)
led_line.request(consumer="LED",
type=gpiod.LINE_REQ_DIR_OUT,
flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP)

try:
	iter = 0
	while iter < 10:
		print(f'iter-{iter}')
		led_line.set_value(0)
		time.sleep(2)
		led_line.set_value(1)
		time.sleep(2)
		iter +=1
finally:
	led_line.release()
