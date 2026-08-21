/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : NUCLEO-F446RE + HW-039/BTS7960 + WGM4632-370
  *                   mechanical-limit stop using HW-039 R_IS / L_IS current sense
  *
  * This firmware does NOT use MT6816.
  *
  * Motor pins:
  *   RPWM -> D5 = PB4 = TIM3_CH1
  *   LPWM -> D9 = PC7 = TIM3_CH2
  *   R_EN -> D7 = PA8
  *   L_EN -> D8 = PA9
  *
  * Current sense:
  *   R_IS -> A0 = PA0 = ADC1_IN0
  *   L_IS -> A1 = PA1 = ADC1_IN1
  *
  * ADC input must be 0~3.3V only.
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>

ADC_HandleTypeDef hadc1;
TIM_HandleTypeDef htim3;
UART_HandleTypeDef huart2;

#define MOTOR_REN_GPIO_Port      GPIOA
#define MOTOR_REN_Pin            GPIO_PIN_8
#define MOTOR_LEN_GPIO_Port      GPIOA
#define MOTOR_LEN_Pin            GPIO_PIN_9
#define MOTOR_PWM_TIMER          htim3
#define MOTOR_RPWM_CHANNEL       TIM_CHANNEL_1
#define MOTOR_LPWM_CHANNEL       TIM_CHANNEL_2

#define PWM_PERIOD_COUNTS        4199U  /* 20 kHz when TIM3 clock is 84 MHz */
#define MANUAL_DUTY_LIMIT_DEFAULT 350
#define MANUAL_DUTY_LIMIT_MAX     900
#define DEFAULT_STALL_THRESHOLD_ADC 1800U
#define DEFAULT_BLANKING_MS        500U
#define DEFAULT_OVERCURRENT_MS     180U
#define DEFAULT_TIMEOUT_MS         5000U

typedef enum {
    STATE_IDLE = 0,
    STATE_MANUAL_FWD,
    STATE_MANUAL_REV,
    STATE_LIMIT_FWD,
    STATE_LIMIT_REV,
    STATE_LIMIT_REACHED_FWD,
    STATE_LIMIT_REACHED_REV,
    STATE_FAULT
} ControlState;

typedef enum {
    SENSE_MAX = 0,
    SENSE_R,
    SENSE_L
} SenseMode;

static ControlState state = STATE_IDLE;
static SenseMode sense_mode = SENSE_MAX;
static int manual_duty_limit = MANUAL_DUTY_LIMIT_DEFAULT;
static int last_signed_duty = 0;
static uint16_t is_r_raw = 0;
static uint16_t is_l_raw = 0;
static uint16_t is_max_raw = 0;
static uint16_t is_used_raw = 0;
static uint16_t stall_threshold_adc = DEFAULT_STALL_THRESHOLD_ADC;
static uint32_t blanking_ms = DEFAULT_BLANKING_MS;
static uint32_t overcurrent_ms = DEFAULT_OVERCURRENT_MS;
static uint32_t move_timeout_ms = DEFAULT_TIMEOUT_MS;
static uint32_t state_start_ms = 0;
static uint32_t over_start_ms = 0;
static bool over_active = false;
static bool auto_status = false;
static uint32_t last_status_ms = 0;
static char fault_msg[128] = "none";
static char rx_buf[96];
static uint8_t rx_len = 0;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM3_Init(void);
static void MX_ADC1_Init(void);

static void uart_print(const char *s);
static void uart_printf(const char *fmt, ...);
static int clamp_int(int x, int lo, int hi);
static uint32_t duty_to_counts(int duty_permille);
static void motor_disable(void);
static void motor_brake(void);
static void motor_forward(int duty_permille);
static void motor_reverse(int duty_permille);
static void motor_set_signed(int signed_duty);
static uint16_t adc_read_channel(uint32_t channel);
static uint16_t raw_to_mv(uint16_t raw);
static uint16_t mv_to_raw(uint16_t mv);
static void is_update(void);
static const char *sense_name(SenseMode m);
static uint16_t sense_value(void);
static void set_state(ControlState s);
static const char *state_name(ControlState s);
static void control_update(void);
static void fault_stop(const char *msg);
static void print_help(void);
static void print_status(void);
static void print_is(void);
static void handle_line(char *line);
static void uart_poll(void);

static void uart_print(const char *s)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)s, strlen(s), HAL_MAX_DELAY);
}

static void uart_printf(const char *fmt, ...)
{
    char buf[320];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    uart_print(buf);
}

static int clamp_int(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static uint32_t duty_to_counts(int duty_permille)
{
    duty_permille = clamp_int(duty_permille, 0, 1000);
    return (uint32_t)(((uint32_t)duty_permille * PWM_PERIOD_COUNTS) / 1000U);
}

static void motor_disable(void)
{
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_RPWM_CHANNEL, 0);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_LPWM_CHANNEL, 0);
    HAL_GPIO_WritePin(MOTOR_REN_GPIO_Port, MOTOR_REN_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(MOTOR_LEN_GPIO_Port, MOTOR_LEN_Pin, GPIO_PIN_RESET);
    last_signed_duty = 0;
}

static void motor_brake(void)
{
    HAL_GPIO_WritePin(MOTOR_REN_GPIO_Port, MOTOR_REN_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(MOTOR_LEN_GPIO_Port, MOTOR_LEN_Pin, GPIO_PIN_SET);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_RPWM_CHANNEL, 0);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_LPWM_CHANNEL, 0);
    last_signed_duty = 0;
}

static void motor_forward(int duty_permille)
{
    duty_permille = clamp_int(duty_permille, 0, 1000);
    HAL_GPIO_WritePin(MOTOR_REN_GPIO_Port, MOTOR_REN_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(MOTOR_LEN_GPIO_Port, MOTOR_LEN_Pin, GPIO_PIN_SET);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_LPWM_CHANNEL, 0);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_RPWM_CHANNEL, duty_to_counts(duty_permille));
    last_signed_duty = duty_permille;
}

static void motor_reverse(int duty_permille)
{
    duty_permille = clamp_int(duty_permille, 0, 1000);
    HAL_GPIO_WritePin(MOTOR_REN_GPIO_Port, MOTOR_REN_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(MOTOR_LEN_GPIO_Port, MOTOR_LEN_Pin, GPIO_PIN_SET);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_RPWM_CHANNEL, 0);
    __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, MOTOR_LPWM_CHANNEL, duty_to_counts(duty_permille));
    last_signed_duty = -duty_permille;
}

static void motor_set_signed(int signed_duty)
{
    signed_duty = clamp_int(signed_duty, -manual_duty_limit, manual_duty_limit);
    if (signed_duty > 0) motor_forward(signed_duty);
    else if (signed_duty < 0) motor_reverse(-signed_duty);
    else motor_brake();
}

static uint16_t adc_read_channel(uint32_t channel)
{
    ADC_ChannelConfTypeDef sConfig = {0};
    sConfig.Channel = channel;
    sConfig.Rank = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) return 0;
    if (HAL_ADC_Start(&hadc1) != HAL_OK) return 0;
    if (HAL_ADC_PollForConversion(&hadc1, 5) != HAL_OK) {
        HAL_ADC_Stop(&hadc1);
        return 0;
    }
    uint16_t value = (uint16_t)HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    return value;
}

static uint16_t raw_to_mv(uint16_t raw)
{
    return (uint16_t)(((uint32_t)raw * 3300U) / 4095U);
}

static uint16_t mv_to_raw(uint16_t mv)
{
    if (mv > 3300U) mv = 3300U;
    return (uint16_t)(((uint32_t)mv * 4095U) / 3300U);
}

static void is_update(void)
{
    is_r_raw = adc_read_channel(ADC_CHANNEL_0);
    is_l_raw = adc_read_channel(ADC_CHANNEL_1);
    is_max_raw = (is_r_raw > is_l_raw) ? is_r_raw : is_l_raw;
    is_used_raw = sense_value();
}

static const char *sense_name(SenseMode m)
{
    switch (m) {
    case SENSE_MAX: return "max";
    case SENSE_R:   return "r";
    case SENSE_L:   return "l";
    default:        return "unknown";
    }
}

static uint16_t sense_value(void)
{
    switch (sense_mode) {
    case SENSE_R:   return is_r_raw;
    case SENSE_L:   return is_l_raw;
    case SENSE_MAX:
    default:        return is_max_raw;
    }
}

static void set_state(ControlState s)
{
    state = s;
    state_start_ms = HAL_GetTick();
    over_start_ms = 0;
    over_active = false;
}

static const char *state_name(ControlState s)
{
    switch (s) {
    case STATE_IDLE:              return "IDLE";
    case STATE_MANUAL_FWD:        return "MANUAL_FWD";
    case STATE_MANUAL_REV:        return "MANUAL_REV";
    case STATE_LIMIT_FWD:         return "LIMIT_FWD";
    case STATE_LIMIT_REV:         return "LIMIT_REV";
    case STATE_LIMIT_REACHED_FWD: return "LIMIT_REACHED_FWD";
    case STATE_LIMIT_REACHED_REV: return "LIMIT_REACHED_REV";
    case STATE_FAULT:             return "FAULT";
    default:                      return "UNKNOWN";
    }
}

static void fault_stop(const char *msg)
{
    motor_disable();
    strncpy(fault_msg, msg, sizeof(fault_msg) - 1);
    fault_msg[sizeof(fault_msg) - 1] = '\0';
    set_state(STATE_FAULT);
    uart_printf("\r\nFAULT: %s\r\n", fault_msg);
}

static void control_update(void)
{
    uint32_t now = HAL_GetTick();
    is_update();

    if (state == STATE_LIMIT_FWD || state == STATE_LIMIT_REV) {
        if ((now - state_start_ms) > move_timeout_ms) {
            fault_stop("limit move timeout");
            return;
        }

        if ((now - state_start_ms) < blanking_ms) {
            over_active = false;
            over_start_ms = 0;
            return;
        }

        uint16_t used = sense_value();
        if (used >= stall_threshold_adc) {
            if (!over_active) {
                over_active = true;
                over_start_ms = now;
            }

            if ((now - over_start_ms) >= overcurrent_ms) {
                motor_brake();
                if (state == STATE_LIMIT_FWD) {
                    set_state(STATE_LIMIT_REACHED_FWD);
                    uart_printf("\r\nFORWARD LIMIT REACHED: IS_%s=%u (%umV), threshold=%u (%umV)\r\n",
                                sense_name(sense_mode), used, raw_to_mv(used),
                                stall_threshold_adc, raw_to_mv(stall_threshold_adc));
                } else {
                    set_state(STATE_LIMIT_REACHED_REV);
                    uart_printf("\r\nREVERSE LIMIT REACHED: IS_%s=%u (%umV), threshold=%u (%umV)\r\n",
                                sense_name(sense_mode), used, raw_to_mv(used),
                                stall_threshold_adc, raw_to_mv(stall_threshold_adc));
                }
                return;
            }
        } else {
            over_active = false;
            over_start_ms = 0;
        }
    }
}

static void print_is(void)
{
    is_update();
    uart_printf("R_IS A0: raw=%u, %umV | L_IS A1: raw=%u, %umV | MAX raw=%u, %umV | used=%s raw=%u, %umV | threshold raw=%u, %umV\r\n",
                is_r_raw, raw_to_mv(is_r_raw),
                is_l_raw, raw_to_mv(is_l_raw),
                is_max_raw, raw_to_mv(is_max_raw),
                sense_name(sense_mode), is_used_raw, raw_to_mv(is_used_raw),
                stall_threshold_adc, raw_to_mv(stall_threshold_adc));
}

static void print_status(void)
{
    is_update();
    uart_printf(
        "\r\nstate=%s duty=%d manual_limit=%d sense=%s R_IS=%u/%umV L_IS=%u/%umV used=%u/%umV threshold=%u/%umV blank=%lums overms=%lums timeout=%lums over_active=%d\r\n",
        state_name(state),
        last_signed_duty,
        manual_duty_limit,
        sense_name(sense_mode),
        is_r_raw, raw_to_mv(is_r_raw),
        is_l_raw, raw_to_mv(is_l_raw),
        is_used_raw, raw_to_mv(is_used_raw),
        stall_threshold_adc, raw_to_mv(stall_threshold_adc),
        (unsigned long)blanking_ms,
        (unsigned long)overcurrent_ms,
        (unsigned long)move_timeout_ms,
        over_active ? 1 : 0
    );
    if (state == STATE_FAULT) uart_printf("fault=%s\r\n", fault_msg);
}

static void print_help(void)
{
    uart_print("\r\n============================================================\r\n");
    uart_print("F446RE + HW-039 R_IS/L_IS limit-stop firmware, no MT6816\r\n");
    uart_print("\r\nPins:\r\n");
    uart_print("  Motor: RPWM->D5, LPWM->D9, R_EN->D7, L_EN->D8\r\n");
    uart_print("  IS   : R_IS->A0, L_IS->A1, GND common\r\n");
    uart_print("  ADC input must be 0~3.3V only.\r\n");
    uart_print("\r\nManual motor commands, no auto-stop:\r\n");
    uart_print("  mf <0..limit>          Manual forward. Example: mf 120\r\n");
    uart_print("  mr <0..limit>          Manual reverse. Example: mr 120\r\n");
    uart_print("  raw <-limit..limit>    Signed manual duty. Example: raw -120\r\n");
    uart_print("  mlimit <0..900>        Set manual duty limit. Default 350\r\n");
    uart_print("\r\nLimit-stop commands, auto-stop by IS:\r\n");
    uart_print("  limf <0..limit>        Forward until IS threshold, then stop\r\n");
    uart_print("  limr <0..limit>        Reverse until IS threshold, then stop\r\n");
    uart_print("\r\nIS current-sense commands:\r\n");
    uart_print("  is                     Print R_IS/L_IS ADC raw and mV\r\n");
    uart_print("  sense max              Use max(R_IS,L_IS), default\r\n");
    uart_print("  sense r                Use only R_IS/A0\r\n");
    uart_print("  sense l                Use only L_IS/A1\r\n");
    uart_print("  thr <0..4095>          Set threshold in ADC raw counts\r\n");
    uart_print("  thrmv <0..3300>        Set threshold in millivolts\r\n");
    uart_print("  blank <ms>             Ignore IS after start. Example: blank 500\r\n");
    uart_print("  overms <ms>            Required over-threshold duration. Example: overms 200\r\n");
    uart_print("  timeout <ms>           Max limit-move duration. Example: timeout 4000\r\n");
    uart_print("\r\nGeneral:\r\n");
    uart_print("  status                 Print full status\r\n");
    uart_print("  auto on/off            Print status every 500 ms\r\n");
    uart_print("  stop                   PWM=0, enable kept high\r\n");
    uart_print("  disable                PWM=0, R_EN/L_EN low\r\n");
    uart_print("  clear                  Clear fault\r\n");
    uart_print("  help                   Show help\r\n");
    uart_print("============================================================\r\n");
}

static void handle_line(char *line)
{
    while (*line == ' ') line++;
    if (strlen(line) == 0) return;

    if (strcmp(line, "help") == 0) {
        print_help();
    } else if (strncmp(line, "mf ", 3) == 0) {
        int duty = clamp_int(atoi(line + 3), 0, manual_duty_limit);
        set_state(STATE_MANUAL_FWD);
        motor_forward(duty);
        uart_printf("MANUAL forward duty=%d/1000. No auto-stop. Type stop to stop.\r\n", duty);
    } else if (strncmp(line, "mr ", 3) == 0) {
        int duty = clamp_int(atoi(line + 3), 0, manual_duty_limit);
        set_state(STATE_MANUAL_REV);
        motor_reverse(duty);
        uart_printf("MANUAL reverse duty=%d/1000. No auto-stop. Type stop to stop.\r\n", duty);
    } else if (strncmp(line, "raw ", 4) == 0) {
        int duty = clamp_int(atoi(line + 4), -manual_duty_limit, manual_duty_limit);
        if (duty > 0) set_state(STATE_MANUAL_FWD);
        else if (duty < 0) set_state(STATE_MANUAL_REV);
        else set_state(STATE_IDLE);
        motor_set_signed(duty);
        uart_printf("MANUAL raw duty=%d/1000. No auto-stop.\r\n", duty);
    } else if (strncmp(line, "limf ", 5) == 0) {
        int duty = clamp_int(atoi(line + 5), 0, manual_duty_limit);
        set_state(STATE_LIMIT_FWD);
        motor_forward(duty);
        uart_printf("LIMIT forward duty=%d/1000. Stop if IS_%s >= %u (%umV) for %lums after %lums blanking.\r\n",
                    duty, sense_name(sense_mode), stall_threshold_adc, raw_to_mv(stall_threshold_adc),
                    (unsigned long)overcurrent_ms, (unsigned long)blanking_ms);
    } else if (strncmp(line, "limr ", 5) == 0) {
        int duty = clamp_int(atoi(line + 5), 0, manual_duty_limit);
        set_state(STATE_LIMIT_REV);
        motor_reverse(duty);
        uart_printf("LIMIT reverse duty=%d/1000. Stop if IS_%s >= %u (%umV) for %lums after %lums blanking.\r\n",
                    duty, sense_name(sense_mode), stall_threshold_adc, raw_to_mv(stall_threshold_adc),
                    (unsigned long)overcurrent_ms, (unsigned long)blanking_ms);
    } else if (strncmp(line, "mlimit ", 7) == 0) {
        manual_duty_limit = clamp_int(atoi(line + 7), 0, MANUAL_DUTY_LIMIT_MAX);
        uart_printf("manual_duty_limit=%d/1000\r\n", manual_duty_limit);
    } else if (strcmp(line, "is") == 0) {
        print_is();
    } else if (strcmp(line, "sense max") == 0) {
        sense_mode = SENSE_MAX;
        uart_print("sense=max\r\n");
    } else if (strcmp(line, "sense r") == 0) {
        sense_mode = SENSE_R;
        uart_print("sense=r, using R_IS/A0 only\r\n");
    } else if (strcmp(line, "sense l") == 0) {
        sense_mode = SENSE_L;
        uart_print("sense=l, using L_IS/A1 only\r\n");
    } else if (strncmp(line, "thr ", 4) == 0) {
        stall_threshold_adc = (uint16_t)clamp_int(atoi(line + 4), 0, 4095);
        uart_printf("stall_threshold_adc=%u, %umV\r\n", stall_threshold_adc, raw_to_mv(stall_threshold_adc));
    } else if (strncmp(line, "thrmv ", 6) == 0) {
        int val = clamp_int(atoi(line + 6), 0, 3300);
        stall_threshold_adc = mv_to_raw((uint16_t)val);
        uart_printf("stall_threshold=%dmV -> raw=%u\r\n", val, stall_threshold_adc);
    } else if (strncmp(line, "blank ", 6) == 0) {
        blanking_ms = (uint32_t)clamp_int(atoi(line + 6), 0, 5000);
        uart_printf("blanking_ms=%lu\r\n", (unsigned long)blanking_ms);
    } else if (strncmp(line, "overms ", 7) == 0) {
        overcurrent_ms = (uint32_t)clamp_int(atoi(line + 7), 10, 3000);
        uart_printf("overcurrent_ms=%lu\r\n", (unsigned long)overcurrent_ms);
    } else if (strncmp(line, "timeout ", 8) == 0) {
        move_timeout_ms = (uint32_t)clamp_int(atoi(line + 8), 100, 60000);
        uart_printf("move_timeout_ms=%lu\r\n", (unsigned long)move_timeout_ms);
    } else if (strcmp(line, "status") == 0) {
        print_status();
    } else if (strcmp(line, "auto on") == 0) {
        auto_status = true;
        uart_print("auto_status=on\r\n");
    } else if (strcmp(line, "auto off") == 0) {
        auto_status = false;
        uart_print("auto_status=off\r\n");
    } else if (strcmp(line, "stop") == 0) {
        motor_brake();
        set_state(STATE_IDLE);
        uart_print("Stopped: PWM=0, enable kept high\r\n");
    } else if (strcmp(line, "disable") == 0) {
        motor_disable();
        set_state(STATE_IDLE);
        uart_print("Disabled: PWM=0, R_EN/L_EN low\r\n");
    } else if (strcmp(line, "clear") == 0) {
        strncpy(fault_msg, "none", sizeof(fault_msg) - 1);
        fault_msg[sizeof(fault_msg) - 1] = '\0';
        motor_brake();
        set_state(STATE_IDLE);
        uart_print("Fault cleared. State=IDLE\r\n");
    } else {
        uart_print("Unknown command. Type help.\r\n");
    }
}

static void uart_poll(void)
{
    uint8_t ch;
    while (HAL_UART_Receive(&huart2, &ch, 1, 0) == HAL_OK) {
        if (ch == '\r' || ch == '\n') {
            if (rx_len > 0) {
                rx_buf[rx_len] = '\0';
                uart_print("\r\n");
                handle_line(rx_buf);
                rx_len = 0;
            }
        } else if (ch == 0x08 || ch == 0x7F) {
            if (rx_len > 0) {
                rx_len--;
                uart_print("\b \b");
            }
        } else {
            if (rx_len < sizeof(rx_buf) - 1) {
                rx_buf[rx_len++] = (char)ch;
                HAL_UART_Transmit(&huart2, &ch, 1, 10);
            }
        }
    }
}

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_TIM3_Init();
    MX_ADC1_Init();

    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
    motor_disable();
    HAL_Delay(300);

    uart_print("\r\nF446RE + HW-039 R_IS/L_IS limit-stop firmware ready.\r\n");
    print_help();
    print_status();

    while (1)
    {
        uint32_t now = HAL_GetTick();
        uart_poll();
        control_update();
        if (auto_status && (now - last_status_ms) >= 500U) {
            last_status_ms = now;
            print_status();
        }
        HAL_Delay(2);
    }
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}

static void MX_ADC1_Init(void)
{
    __HAL_RCC_ADC1_CLK_ENABLE();
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode = DISABLE;
    hadc1.Init.ContinuousConvMode = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    hadc1.Init.DMAContinuousRequests = DISABLE;
    hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    if (HAL_ADC_Init(&hadc1) != HAL_OK) Error_Handler();
}

static void MX_TIM3_Init(void)
{
    __HAL_RCC_TIM3_CLK_ENABLE();
    TIM_ClockConfigTypeDef sClockSourceConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};
    TIM_OC_InitTypeDef sConfigOC = {0};

    htim3.Instance = TIM3;
    htim3.Init.Prescaler = 0;
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = PWM_PERIOD_COUNTS;
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim3) != HAL_OK) Error_Handler();

    sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
    if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK) Error_Handler();
    if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) Error_Handler();

    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK) Error_Handler();

    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 0;
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) Error_Handler();
}

static void MX_USART2_UART_Init(void)
{
    __HAL_RCC_USART2_CLK_ENABLE();
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8 | GPIO_PIN_9, GPIO_PIN_RESET);

    /* A0 = PA0 = R_IS, A1 = PA1 = L_IS */
    GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* USART2 VCP: PA2 TX, PA3 RX */
    GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* D7 = PA8, D8 = PA9 */
    GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* D5 = PB4 = TIM3_CH1 */
    GPIO_InitStruct.Pin = GPIO_PIN_4;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF2_TIM3;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* D9 = PC7 = TIM3_CH2 */
    GPIO_InitStruct.Pin = GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF2_TIM3;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) { }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    (void)file;
    (void)line;
}
#endif
