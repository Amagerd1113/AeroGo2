-- AeroGo2 two-key arming gate for ArduCopter 4.6+
--
-- Key 1: AeroGo2 shell sends custom MAV_CMD 31000 authorization heartbeats.
-- Key 2: the RadioMaster flight-enable switch on RC channel 5 must then move
--        from LOW to HIGH.
--
-- This script never force-arms.  It blocks MAVLink Arm (including force-Arm),
-- preserves checked MAVLink Disarm, keeps ArduPilot's auxiliary arming check
-- failed until both keys arrive in order, then invokes arming:arm(), which runs
-- the normal ArduPilot arming checks.

local mavlink_msgs = require("MAVLink/mavlink_msgs")

local COMMAND_LONG_ID = mavlink_msgs.get_msgid("COMMAND_LONG")

local AEROGO2_AUTH_COMMAND = 31000
local AEROGO2_AUTH_MAGIC = 6202
local AEROGO2_AUTH_PROTOCOL = 1
local AUTH_HEARTBEAT_TIMEOUT_MS = 1500
local MAV_CMD_COMPONENT_ARM_DISARM = 400
local UPDATE_PERIOD_MS = 50
local ARM_SWITCH_CHANNEL = 5
local ARM_SWITCH_LOW_MAX = 1200
local ARM_SWITCH_HIGH_MIN = 1800

local MAV_RESULT_ACCEPTED = 0
local MAV_RESULT_TEMPORARILY_REJECTED = 1
local MAV_RESULT_DENIED = 2
local MAV_RESULT_FAILED = 4
local MAV_SEVERITY_WARNING = 4
local MAV_SEVERITY_NOTICE = 5
local MAV_SEVERITY_INFO = 6

local auth_id = arming:get_aux_auth_id()
if auth_id == nil then
    gcs:send_text(MAV_SEVERITY_WARNING, "AeroGo2 arm gate: no AuxAuth ID")
    error("AeroGo2 arm gate cannot reserve AuxAuth")
end

mavlink:init(10, 1)
mavlink:register_rx_msgid(COMMAND_LONG_ID)
mavlink:block_command(AEROGO2_AUTH_COMMAND)
mavlink:block_command(MAV_CMD_COMPONENT_ARM_DISARM)

local message_map = {}
message_map[COMMAND_LONG_ID] = "COMMAND_LONG"

local authorized = false
local last_authorization_ms = uint32_t(0)
local last_failure_reason = nil
local last_arm_switch_high = false

local function set_failed(reason)
    if last_failure_reason ~= reason then
        arming:set_aux_auth_failed(auth_id, reason)
        last_failure_reason = reason
    end
end

local function send_ack(channel, command, result, sequence, target_system, target_component)
    local ack = {}
    ack.command = command
    ack.result = result
    ack.progress = 0
    ack.result_param2 = sequence
    ack.target_system = target_system
    ack.target_component = target_component
    mavlink:send_chan(channel, mavlink_msgs.encode("COMMAND_ACK", ack))
end

local function revoke(reason)
    authorized = false
    last_authorization_ms = uint32_t(0)
    set_failed(reason)
end

local function valid_arm_switch_low()
    if not rc:has_valid_input() then
        return false, "AeroGo2: RC input invalid"
    end
    local pwm = rc:get_pwm(ARM_SWITCH_CHANNEL)
    if pwm == nil then
        return false, "AeroGo2: RC5 missing"
    end
    if pwm > ARM_SWITCH_LOW_MAX then
        return false, "AeroGo2: set RC5 LOW first"
    end
    return true, nil
end

local function valid_gate_parameters()
    local required = {
        {"RC5_OPTION", 153},
        {"ARMING_RUDDER", 0},
        {"ARMING_CHECK", 1},
    }
    for i = 1, #required do
        local name = required[i][1]
        local expected = required[i][2]
        local actual = param:get(name)
        if actual == nil or math.floor(actual) ~= expected then
            return false, "AeroGo2: set " .. name .. "=" .. tostring(expected)
        end
    end
    local skipped = param:get("ARMING_SKIPCHK")
    if skipped ~= nil and math.floor(skipped) ~= 0 then
        return false, "AeroGo2: set ARMING_SKIPCHK=0"
    end
    return true, nil
end

local function handle_authorization(command)
    local sequence = math.floor(command.param4 or 0)
    local target_system = command.sysid or 0
    local target_component = command.compid or 0

    if math.floor(command.param2 or 0) ~= AEROGO2_AUTH_MAGIC or
       math.floor(command.param3 or 0) ~= AEROGO2_AUTH_PROTOCOL then
        return MAV_RESULT_DENIED, sequence, target_system, target_component
    end

    if (command.param1 or 0) < 0.5 then
        revoke("AeroGo2: ground authorization required")
        gcs:send_text(MAV_SEVERITY_INFO, "AeroGo2 ground authorization revoked")
        return MAV_RESULT_ACCEPTED, sequence, target_system, target_component
    end

    local params_ok, params_reason = valid_gate_parameters()
    if not params_ok then
        revoke(params_reason)
        gcs:send_text(MAV_SEVERITY_WARNING, params_reason)
        return MAV_RESULT_TEMPORARILY_REJECTED, sequence, target_system, target_component
    end

    if arming:is_armed() then
        return MAV_RESULT_DENIED, sequence, target_system, target_component
    end

    if not authorized then
        local switch_low, reason = valid_arm_switch_low()
        if not switch_low then
            revoke(reason)
            return MAV_RESULT_TEMPORARILY_REJECTED, sequence, target_system, target_component
        end
        authorized = true
        last_arm_switch_high = false
        set_failed("AeroGo2: waiting for RadioMaster RC5 HIGH")
        gcs:send_text(MAV_SEVERITY_NOTICE, "AeroGo2 ground authorized; raise RC5")
    elseif not rc:has_valid_input() then
        revoke("AeroGo2: RC input invalid")
        return MAV_RESULT_TEMPORARILY_REJECTED, sequence, target_system, target_component
    end

    last_authorization_ms = millis()
    return MAV_RESULT_ACCEPTED, sequence, target_system, target_component
end

local function handle_blocked_arm_disarm(command)
    local target_system = command.sysid or 0
    local target_component = command.compid or 0

    if (command.param1 or 0) < 0.5 then
        local accepted = arming:disarm()
        if accepted then
            gcs:send_text(MAV_SEVERITY_NOTICE, "AeroGo2 checked MAVLink DISARM accepted")
            return MAV_RESULT_ACCEPTED, 0, target_system, target_component
        end
        return MAV_RESULT_FAILED, 0, target_system, target_component
    end

    gcs:send_text(
        MAV_SEVERITY_WARNING,
        "AeroGo2 blocked MAVLink ARM; use shell authorize then RC5"
    )
    return MAV_RESULT_DENIED, 0, target_system, target_component
end

local function receive_commands()
    for _ = 1, 10 do
        local raw_message, channel = mavlink:receive_chan()
        if raw_message == nil then
            return
        end
        local command = mavlink_msgs.decode(raw_message, message_map)
        if command ~= nil and command.msgid == COMMAND_LONG_ID then
            local result = nil
            local sequence = 0
            local target_system = command.sysid or 0
            local target_component = command.compid or 0
            if command.command == AEROGO2_AUTH_COMMAND then
                result, sequence, target_system, target_component =
                    handle_authorization(command)
            elseif command.command == MAV_CMD_COMPONENT_ARM_DISARM then
                result, sequence, target_system, target_component =
                    handle_blocked_arm_disarm(command)
            end
            if result ~= nil then
                send_ack(
                    channel,
                    command.command,
                    result,
                    sequence,
                    target_system,
                    target_component
                )
            end
        end
    end
end

local function attempt_checked_arm()
    authorized = false
    arming:set_aux_auth_passed(auth_id)
    last_failure_reason = nil
    local accepted = arming:arm()
    if accepted then
        gcs:send_text(MAV_SEVERITY_NOTICE, "AeroGo2 two-key ARM accepted")
        set_failed("AeroGo2: authorization consumed")
    else
        gcs:send_text(MAV_SEVERITY_WARNING, "AeroGo2 ARM rejected by pre-arm checks")
        set_failed("AeroGo2: ARM rejected; authorize again")
    end
end

local function update()
    receive_commands()

    if arming:is_armed() then
        authorized = false
        set_failed("AeroGo2: authorization consumed")
        return update, UPDATE_PERIOD_MS
    end

    if not authorized then
        set_failed("AeroGo2: ground authorization required")
        return update, UPDATE_PERIOD_MS
    end

    if millis() - last_authorization_ms > AUTH_HEARTBEAT_TIMEOUT_MS then
        revoke("AeroGo2: ground authorization timeout")
        gcs:send_text(MAV_SEVERITY_WARNING, "AeroGo2 authorization timeout")
        return update, UPDATE_PERIOD_MS
    end

    if not rc:has_valid_input() then
        revoke("AeroGo2: RC input invalid")
        return update, UPDATE_PERIOD_MS
    end

    local pwm = rc:get_pwm(ARM_SWITCH_CHANNEL)
    if pwm == nil then
        revoke("AeroGo2: RC5 missing")
        return update, UPDATE_PERIOD_MS
    end

    local switch_high = pwm >= ARM_SWITCH_HIGH_MIN
    if switch_high and not last_arm_switch_high then
        last_arm_switch_high = true
        attempt_checked_arm()
    elseif pwm <= ARM_SWITCH_LOW_MAX then
        last_arm_switch_high = false
    end

    return update, UPDATE_PERIOD_MS
end

set_failed("AeroGo2: ground authorization required")
gcs:send_text(MAV_SEVERITY_INFO, "AeroGo2 two-key arm gate loaded")
return update()
