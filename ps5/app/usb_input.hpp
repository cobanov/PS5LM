// PS5 USB keyboard, mouse, and HDMI-CEC media remote input alongside DualSense.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/input.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ps5lm {

// ---- Raw PS5/Orbis ABI structs (host-tested via static_asserts) -----------

struct RawKeyboardOpenParam {
    std::uint8_t behavior_flag = 0;
    std::uint8_t reserved[7]   = {};
};
static_assert(sizeof(RawKeyboardOpenParam) == 8);

// One reading as PS5 libSceKeyboard.sprx writes it (96 bytes / 0x60, one per
// key or connection change; each reading lists every HID key currently held
// down, followed by 32 trailing reserved bytes).
struct RawKeyboardSample {
    std::uint64_t timestamp_us  = 0;   // 0x00
    std::uint8_t  intercepted   = 0;   // 0x08 (nonzero when system UI owns keyboard focus)
    std::uint8_t  padding0[7]   = {};  // 0x09
    std::uint8_t  connected     = 0;   // 0x10 (nonzero when a keyboard is attached)
    std::uint8_t  padding1[3]   = {};  // 0x11
    std::uint32_t count         = 0;   // 0x14 (keys down, and 1 with none)
    std::uint32_t locks         = 0;   // 0x18 (1=NumLock, 2=CapsLock, 4=ScrollLock)
    std::uint32_t modifiers     = 0;   // 0x1C (0x01 LCtrl .. 0x80 RGui)
    std::uint16_t keycodes[16]  = {};  // 0x20 (active USB HID Usage Page 0x07 codes)
    std::uint8_t  reserved2[32] = {};  // 0x40 (trailing 32 bytes in 0x60-byte ABI)
};
static_assert(sizeof(RawKeyboardSample) == 96);
static_assert(alignof(RawKeyboardSample) == 8);
static_assert(offsetof(RawKeyboardSample, intercepted) == 0x08);
static_assert(offsetof(RawKeyboardSample, connected) == 0x10);
static_assert(offsetof(RawKeyboardSample, count) == 0x14);
static_assert(offsetof(RawKeyboardSample, locks) == 0x18);
static_assert(offsetof(RawKeyboardSample, modifiers) == 0x1C);
static_assert(offsetof(RawKeyboardSample, keycodes) == 0x20);
static_assert(offsetof(RawKeyboardSample, reserved2) == 0x40);

struct KeyboardPortSamples {
    int                                port = 0;
    std::span<const RawKeyboardSample> samples;
};

struct RawMouseOpenParam {
    std::uint8_t behavior_flag = 0;
    std::uint8_t reserved[7]   = {};
};
static_assert(sizeof(RawMouseOpenParam) == 8);

struct RawMouseData {
    std::uint64_t timestamp_us = 0;
    std::int32_t  connected    = 0;
    std::uint32_t buttons      = 0;  // 0x01 L, 0x02 R, 0x04 M, 0x08 Back, 0x10 Fwd
    std::int32_t  x_axis       = 0;  // relative delta X
    std::int32_t  y_axis       = 0;  // relative delta Y
    std::int32_t  wheel        = 0;  // >0 scroll up, <0 scroll down
    std::int32_t  tilt         = 0;
    std::uint64_t reserved     = 0;
};
static_assert(sizeof(RawMouseData) == 40);
static_assert(offsetof(RawMouseData, connected) == 0x08);
static_assert(offsetof(RawMouseData, buttons) == 0x0C);
static_assert(offsetof(RawMouseData, x_axis) == 0x10);
static_assert(offsetof(RawMouseData, y_axis) == 0x14);
static_assert(offsetof(RawMouseData, wheel) == 0x18);
static_assert(offsetof(RawMouseData, tilt) == 0x1C);

struct MousePortSamples {
    int                           port = 0;
    std::span<const RawMouseData> samples;
};

struct RawRemoteSample {
    std::uint32_t buttons   = 0;
    bool          connected = false;
    std::uint8_t  cec_code  = 0;
};

namespace keyboard_bits {
constexpr std::uint32_t kNumLock    = 0x01;
constexpr std::uint32_t kCapsLock   = 0x02;
constexpr std::uint32_t kScrollLock = 0x04;

constexpr std::uint32_t kLeftCtrl   = 0x01;
constexpr std::uint32_t kLeftShift  = 0x02;
constexpr std::uint32_t kLeftAlt    = 0x04;
constexpr std::uint32_t kLeftGui    = 0x08;
constexpr std::uint32_t kRightCtrl  = 0x10;
constexpr std::uint32_t kRightShift = 0x20;
constexpr std::uint32_t kRightAlt   = 0x40;
constexpr std::uint32_t kRightGui   = 0x80;
}  // namespace keyboard_bits

namespace mouse_bits {
constexpr std::uint32_t kLeft        = 0x00000001u;
constexpr std::uint32_t kRight       = 0x00000002u;
constexpr std::uint32_t kMiddle      = 0x00000004u;
constexpr std::uint32_t kBack        = 0x00000008u;
constexpr std::uint32_t kForward     = 0x00000010u;
constexpr std::uint32_t kIntercepted = 0x80000000u;
}  // namespace mouse_bits

// ---- High-level events consumed by Dashboard -------------------------------

struct TextInput {
    std::string typed;
    int         backspaces = 0;
    bool        submit     = false;
    bool        cancel     = false;

    bool empty() const { return typed.empty() && backspaces == 0 && !submit && !cancel; }
};

struct PointerInput {
    float x              = 960.0f;
    float y              = 540.0f;
    bool  moved          = false;
    bool  visible        = false;
    bool  left_down      = false;
    bool  left_pressed   = false;
    bool  left_released  = false;
    bool  right_pressed  = false;
    bool  middle_pressed = false;
    int   wheel          = 0;
};

enum class InputMode : std::uint8_t {
    controller,
    keyboard_mouse,
};

struct InputActivity {
    bool pad_connected = false;
    bool pad_active    = false;
    bool usb_connected = false;
    bool usb_active    = false;
};

struct UsbFrame {
    hui::InputFrame actions;
    TextInput       text;
    PointerInput    pointer;
    bool            keyboard_active    = false;
    bool            mouse_active       = false;
    bool            keyboard_connected = false;
    bool            mouse_connected    = false;
    bool            keyboard_activity  = false;
    bool            mouse_activity     = false;
};

// Decodes a USB HID Usage Page 0x07 scancode with US ANSI layout to ASCII, or
// '\0' when non-printable or Ctrl/Alt/Meta is held.
char hid_scancode_to_ascii(std::uint16_t scancode, std::uint32_t modifiers, std::uint32_t locks);

class KeyboardTracker {
  public:
    static constexpr int kMaxPorts = 8;

    void update(std::span<const KeyboardPortSamples> port_batches, std::uint64_t now_us, bool text_mode,
                hui::InputFrame * out_actions, TextInput * out_text, bool * out_activity = nullptr);

    void set_port_connected(int port, bool connected);
    void clear_port(int port);
    bool connected() const { return connected_; }

  private:
    struct PortState {
        bool                  connected = false;
        std::uint32_t         modifiers = 0;
        std::uint32_t         locks     = 0;
        std::array<bool, 256> down{};
    };

    std::array<PortState, kMaxPorts> ports_{};
    std::array<bool, 256>            down_{};
    bool                             connected_        = false;
    bool                             prev_text_mode_   = false;
    bool                             locks_seen_       = false;
    bool                             soft_caps_lock_   = false;
    std::uint32_t                    prev_modifiers_   = 0;
    std::uint32_t                    held_actions_     = 0;
    hui::Direction                   nav_held_         = hui::Direction::none;
    std::uint64_t                    nav_next_us_      = 0;
    std::uint16_t                    repeat_code_      = 0;
    bool                             repeat_backspace_ = false;
    std::uint64_t                    repeat_next_us_   = 0;
};

class MouseTracker {
  public:
    static constexpr int           kMaxPorts          = 4;
    static constexpr float         kSensitivity       = 1.5f;
    static constexpr std::uint64_t kCursorHideDelayUs = 5'000'000;

    void update(std::span<const MousePortSamples> port_batches, std::uint64_t now_us, bool hide_cursor,
                hui::InputFrame * out_actions, PointerInput * out_pointer);

    // Compatibility single-port wrapper; templated so empty `{}` calls resolve
    // unambiguously to the multi-port overload.
    template <int = 0>
    void update(std::span<const RawMouseData> packets, std::uint64_t now_us, bool hide_cursor,
                hui::InputFrame * out_actions, PointerInput * out_pointer) {
        if (packets.empty()) {
            update(std::span<const MousePortSamples>{}, now_us, hide_cursor, out_actions, out_pointer);
            return;
        }
        const MousePortSamples batch{ 0, packets };
        update(std::span<const MousePortSamples>(&batch, 1), now_us, hide_cursor, out_actions, out_pointer);
    }

    void  clear_port(int port);
    bool  connected() const { return connected_; }
    float x() const { return x_; }
    float y() const { return y_; }
    bool  visible() const { return visible_; }

  private:
    struct PortState {
        bool          connected = false;
        std::uint32_t buttons   = 0;
    };

    std::array<PortState, kMaxPorts> ports_{};
    float                            x_                = 960.0f;
    float                            y_                = 540.0f;
    bool                             visible_          = false;
    bool                             connected_        = false;
    std::uint32_t                    held_buttons_     = 0;
    std::uint32_t                    held_actions_     = 0;
    std::uint64_t                    last_activity_us_ = 0;
};
static_assert(MouseTracker::kMaxPorts == 4);

class RemoteTracker {
  public:
    hui::InputFrame update(const RawRemoteSample & sample, std::uint64_t now_us);

  private:
    std::uint32_t  held_actions_ = 0;
    hui::Direction nav_held_     = hui::Direction::none;
    std::uint64_t  nav_next_us_  = 0;
    bool           connected_    = false;
};

// Combines DualSense input with USB/remote input so either controller or
// keyboard/mouse/remote can drive the UI seamlessly.
hui::InputFrame merge_input(const hui::InputFrame & pad, const hui::InputFrame & usb);

// Live PS5 USB + HDMI-CEC input runtime (safe no-op on non-PS5 host builds).
class UsbInput {
  public:
    UsbInput() = default;
    ~UsbInput();

    UsbInput(const UsbInput &) = delete;
    UsbInput & operator=(const UsbInput &) = delete;

    void     open();
    void     close();
    UsbFrame poll(std::uint64_t now_us, bool text_mode, bool pad_activity);

    bool keyboard_ready() const { return kbd_ready_; }
    bool mouse_ready() const { return mouse_ready_; }
    bool remote_ready() const { return remote_handle_ >= 0; }

  private:
    void refresh_devices(std::uint64_t now_us);

    struct KeyboardSlot {
        int           user_id       = -1;
        int           index         = 0;
        int           handle        = -1;
        bool          logged_conn   = false;
        bool          conn_state    = false;
        std::uint32_t last_read_err = 0;
    };
    struct MouseSlot {
        int user_id = -1;
        int handle  = -1;
    };

    bool kbd_ready_     = false;
    bool mouse_ready_   = false;
    int  remote_handle_ = -1;
#ifdef __PROSPERO__
    bool          opened_          = false;
    bool          kbd_logged_on_   = false;
    bool          mouse_logged_on_ = false;
    std::uint64_t last_refresh_us_ = 0;

    std::array<KeyboardSlot, KeyboardTracker::kMaxPorts> kbd_slots_{};
    std::array<MouseSlot, MouseTracker::kMaxPorts>       mouse_slots_{};
#endif

    KeyboardTracker kbd_tracker_;
    MouseTracker    mouse_tracker_;
    RemoteTracker   remote_tracker_;
};

}  // namespace ps5lm
