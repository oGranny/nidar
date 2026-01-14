from pymavlink import mavutil
import time
import json
import os

# Load servo constants from config
# SERVO_MIN_ANGLE = config.get('SRV_MIN_ANG')      
# SERVO_MAX_ANGLE = config.get('SRV_MAX_ANG')
# SERVO_MIN_PWM = config.get('SRV_MIN_PWM')
# SERVO_MAX_PWM = config.get('SRV_MAX_PWM')

# def angle_to_pwm(angle_deg: float,
#                  *, min_angle: float = SERVO_MIN_ANGLE,
#                  max_angle: float = SERVO_MAX_ANGLE,
#                  min_pwm: int = SERVO_MIN_PWM,
#                  max_pwm: int = SERVO_MAX_PWM) -> int:
#     angle = max(min(angle_deg, max_angle), min_angle)
#     span_a = max_angle - min_angle if max_angle != min_angle else 1.0
#     span_p = max_pwm - min_pwm
#     pwm = int(round(min_pwm + (angle - min_angle) * (span_p / span_a)))
#     return max(min(pwm, max_pwm), min_pwm)

def set_servo_pwm(master, config, servo_output: int, pwm: int):
    print(f"Sending SERVO{servo_output} -> PWM {pwm}")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        float(servo_output),  # param1: servo number
        float(pwm),           # param2: PWM value
        0, 0, 0, 0, 0
    )

def reset_servos(master, config):
    locks = config.get("SRV_PWM_LOCK")
    for key, val in locks.items():
        set_servo_pwm(master, config, int(key),   int(val))

def drop_packet(master, config, servo_output):
    map = config.get("SRV_PWM_OPEN")
    set_servo_pwm(master, config, servo_output, int(map[str(servo_output)]))
    set_servo_pwm(master, config, config.get("BUZZ_PWM_PIN"), int(config.get("BUZZ_PWM_LOCK")))
    time.sleep(config.get("SLP_AFT_BUZZ", 2.0))
    set_servo_pwm(master, config, config.get("BUZZ_PWM_PIN"), int(0))
    time.sleep(config.get("SLP_AFT_DRP", 1.0))
