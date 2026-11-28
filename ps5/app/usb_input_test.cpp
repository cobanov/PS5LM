// Unit tests for PS5 USB keyboard, mouse, and HDMI-CEC remote input tracking.
//
// Build and run (no PS5 SDK required):
//   c++ -std=c++20 -O2 -Wall -Wextra -Ithird_party/ps5-homebrew-ui/src -Ips5/app ps5/app/usb_input_test.cpp ps5/app/usb_input.cpp third_party/ps5-homebrew-ui/src/core/input.cpp -o build/usb_input_test && build/usb_input_test
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "usb_input.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {

using ps5lm::KeyboardPortSamples;
using ps5lm::KeyboardTracker;
using ps5lm::MousePortSamples;
using ps5lm::MouseTracker;
using ps5lm::PointerInput;
using ps5lm::RawKeyboardSample;
using ps5lm::RawMouseData;
using ps5lm::RawRemoteSample;
using ps5lm::RemoteTracker;
using ps5lm::TextInput;
using ps5lm::hid_scancode_to_ascii;
using ps5lm::merge_input;

RawKeyboardSample make_kbd(bool connected, std::initializer_list<std::uint16_t> keys,
                           std::uint32_t modifiers = 0, std::uint32_t locks = 0) {
    RawKeyboardSample s{};
    s.connected = connected ? 1u : 0u;
    s.modifiers = modifiers;
    s.locks     = locks;
    std::uint32_t idx = 0;
    for (std::uint16_t k : keys) {
        if (idx < 16) {
            s.keycodes[idx++] = k;
        }
    }
    s.count = idx > 0 ? idx : 1u;
    return s;
}

void feed_one(KeyboardTracker * kt, int port, const RawKeyboardSample & s, std::uint64_t now_us, bool text_mode,
              hui::InputFrame * act, TextInput * txt, bool * activity = nullptr) {
    KeyboardPortSamples batch{ port, std::span<const RawKeyboardSample>(&s, 1) };
    kt->update(std::span<const KeyboardPortSamples>(&batch, 1), now_us, text_mode, act, txt, activity);
}

void test_hid_decoding() {
    assert(hid_scancode_to_ascii(0x04, 0, 0) == 'a');
    assert(hid_scancode_to_ascii(0x1D, 0, 0) == 'z');
    assert(hid_scancode_to_ascii(0x04, ps5lm::keyboard_bits::kLeftShift, 0) == 'A');
    assert(hid_scancode_to_ascii(0x04, 0, ps5lm::keyboard_bits::kCapsLock) == 'A');
    assert(hid_scancode_to_ascii(0x04, ps5lm::keyboard_bits::kRightShift, ps5lm::keyboard_bits::kCapsLock) == 'a');

    assert(hid_scancode_to_ascii(0x1E, 0, 0) == '1');
    assert(hid_scancode_to_ascii(0x1E, ps5lm::keyboard_bits::kLeftShift, 0) == '!');
    assert(hid_scancode_to_ascii(0x27, ps5lm::keyboard_bits::kLeftShift, 0) == ')');

    assert(hid_scancode_to_ascii(0x2C, 0, 0) == ' ');
    assert(hid_scancode_to_ascii(0x2D, 0, 0) == '-');
    assert(hid_scancode_to_ascii(0x2D, ps5lm::keyboard_bits::kLeftShift, 0) == '_');
    assert(hid_scancode_to_ascii(0x38, ps5lm::keyboard_bits::kRightShift, 0) == '?');
    assert(hid_scancode_to_ascii(0x64, 0, 0) == '\\');
    assert(hid_scancode_to_ascii(0x64, ps5lm::keyboard_bits::kLeftShift, 0) == '|');

    assert(hid_scancode_to_ascii(0x54, 0, 0) == '/');
    assert(hid_scancode_to_ascii(0x57, 0, 0) == '+');
    assert(hid_scancode_to_ascii(0x59, 0, 0) == '1');
    assert(hid_scancode_to_ascii(0x62, 0, 0) == '0');
    assert(hid_scancode_to_ascii(0x63, 0, 0) == '.');

    // Ctrl / Alt / Meta suppress ASCII emission.
    assert(hid_scancode_to_ascii(0x04, ps5lm::keyboard_bits::kLeftCtrl, 0) == '\0');
    assert(hid_scancode_to_ascii(0x04, ps5lm::keyboard_bits::kLeftAlt, 0) == '\0');
    assert(hid_scancode_to_ascii(0x04, ps5lm::keyboard_bits::kLeftGui, 0) == '\0');
}

void test_keyboard_nav_and_rollover() {
    KeyboardTracker kt;
    hui::InputFrame act;
    TextInput       txt;
    bool            activity = false;

    // Arrow down press -> nav down once, then repeat on 0-sample idle frames after 350ms + 110ms.
    RawKeyboardSample down_s = make_kbd(true, { 0x51 });
    feed_one(&kt, 0, down_s, 1'000'000, false, &act, &txt, &activity);
    assert(act.connected);
    assert(activity);
    assert(act.is_pressed(hui::Action::down));
    assert(act.nav == hui::Direction::down && !act.nav_repeat);

    // 0-sample frame before repeat delay: held key & connected stay alive, no repeat, no activity.
    kt.update({}, 1'200'000, false, &act, &txt, &activity);
    assert(act.connected);
    assert(act.is_held(hui::Action::down));
    assert(act.nav == hui::Direction::none);
    assert(!activity);

    // 0-sample frames past repeat delay & interval fire nav repeat and report activity.
    kt.update({}, 1'360'000, false, &act, &txt, &activity);
    assert(act.nav == hui::Direction::down && act.nav_repeat);
    assert(activity);

    kt.update({}, 1'475'000, false, &act, &txt, &activity);
    assert(act.nav == hui::Direction::down && act.nav_repeat);
    assert(activity);

    // Slot-shift rollover test: Enter+Space in slots [0],[1], then Enter released so
    // Space shifts into slot [0]. confirm must remain held without spurious re-press.
    RawKeyboardSample both = make_kbd(true, { 0x28, 0x2C });
    feed_one(&kt, 0, both, 1'500'000, false, &act, &txt);
    assert(act.is_pressed(hui::Action::confirm));
    assert(act.is_held(hui::Action::confirm));

    RawKeyboardSample only_space_shifted = make_kbd(true, { 0x2C });
    feed_one(&kt, 0, only_space_shifted, 1'516'000, false, &act, &txt);
    assert(!act.is_pressed(hui::Action::confirm));
    assert(act.is_held(hui::Action::confirm));

    // All nav bindings work.
    struct {
        std::uint16_t code;
        std::uint32_t mods;
        hui::Action   expected;
    } cases[] = {
        { 0x29, 0, hui::Action::back },       // Escape
        { 0x2A, 0, hui::Action::back },       // Backspace
        { 0x2B, 0, hui::Action::page_next },  // Tab
        { 0x2B, ps5lm::keyboard_bits::kLeftShift, hui::Action::page_prev },
        { 0x4B, 0, hui::Action::page_prev },  // PageUp
        { 0x4E, 0, hui::Action::page_next },  // PageDown
        { 0x36, 0, hui::Action::jump_prev },  // Comma
        { 0x37, 0, hui::Action::jump_next },  // Period
        { 0x1B, 0, hui::Action::west },       // X -> Square
        { 0x17, 0, hui::Action::north },      // T -> Triangle
        { 0x43, 0, hui::Action::menu },       // F10
    };
    RawKeyboardSample idle = make_kbd(true, {});
    for (const auto & c : cases) {
        feed_one(&kt, 0, idle, 2'000'000, false, &act, &txt);
        RawKeyboardSample k = make_kbd(true, { c.code }, c.mods);
        feed_one(&kt, 0, k, 2'016'000, false, &act, &txt);
        assert(act.is_pressed(c.expected));
    }

    // Disconnect releases cleanly.
    RawKeyboardSample disc = make_kbd(false, {});
    feed_one(&kt, 0, disc, 2'100'000, false, &act, &txt);
    assert(!act.connected);
    assert(!kt.connected());
    assert(act.focus_lost);
    assert(act.held == 0);
}

void test_keyboard_text_mode_and_transition_guard() {
    KeyboardTracker kt;
    hui::InputFrame act;
    TextInput       txt;
    bool            activity = false;

    // Press T on Home to open prompt; still held on first frame of text_mode (0-sample frame) -> no 't' typed.
    RawKeyboardSample t_down = make_kbd(true, { 0x17 });
    feed_one(&kt, 0, t_down, 1'000'000, false, &act, &txt);
    assert(act.is_pressed(hui::Action::north));
    assert(txt.empty());

    kt.update({}, 1'016'000, true, &act, &txt, &activity);
    assert(txt.empty());
    assert(act.pressed == 0);
    assert(!activity);

    // Release keys, then type "Hi" and test 0-sample hold-repeat for chars & backspace.
    RawKeyboardSample idle = make_kbd(true, {});
    feed_one(&kt, 0, idle, 1'032'000, true, &act, &txt);

    RawKeyboardSample shift_h = make_kbd(true, { 0x0B }, ps5lm::keyboard_bits::kLeftShift);
    feed_one(&kt, 0, shift_h, 1'050'000, true, &act, &txt, &activity);
    assert(txt.typed == "H");
    assert(act.pressed == 0);
    assert(activity);

    RawKeyboardSample i_key = make_kbd(true, { 0x0C });
    feed_one(&kt, 0, i_key, 1'070'000, true, &act, &txt, &activity);
    assert(txt.typed == "i");
    assert(activity);

    // Hold 'i' across 0-sample frame past 400ms repeat delay -> repeats 'i'.
    kt.update({}, 1'480'000, true, &act, &txt, &activity);
    assert(txt.typed == "i");
    assert(activity);

    // Backspace once, then hold across 0-sample frame past 400ms for auto-repeat.
    RawKeyboardSample bs = make_kbd(true, { 0x2A });
    feed_one(&kt, 0, bs, 1'500'000, true, &act, &txt, &activity);
    assert(txt.backspaces == 1);
    assert(activity);
    kt.update({}, 1'910'000, true, &act, &txt, &activity);
    assert(txt.backspaces == 1);
    assert(activity);

    // Non-nav keys in text_mode do not fire page/back actions; Esc sets txt.cancel.
    RawKeyboardSample esc = make_kbd(true, { 0x29 });
    feed_one(&kt, 0, esc, 2'000'000, true, &act, &txt);
    assert(txt.cancel);
    assert(!act.is_pressed(hui::Action::back));

    // Held Esc exiting text_mode into nav mode (0-sample frame) must NOT double-trigger Action::back.
    kt.update({}, 2'016'000, false, &act, &txt);
    assert(!act.is_pressed(hui::Action::back));
    assert(act.is_held(hui::Action::back));

    feed_one(&kt, 0, idle, 2'032'000, false, &act, &txt);
    RawKeyboardSample ret = make_kbd(true, { 0x28 });
    feed_one(&kt, 0, ret, 2'100'000, true, &act, &txt);
    assert(txt.submit);
    kt.update({}, 2'116'000, false, &act, &txt);
    assert(!act.is_pressed(hui::Action::confirm));
    assert(act.is_held(hui::Action::confirm));
}

void test_keyboard_fast_taps_multiport_errors_and_capslock() {
    KeyboardTracker kt;
    hui::InputFrame act;
    TextInput       txt;
    bool            activity = false;

    // 1. Seed port 0 connected (as sceKeyboardGetConnection does on boot) -> connected without activity.
    kt.set_port_connected(0, true);
    kt.update({}, 1'000'000, false, &act, &txt, &activity);
    assert(kt.connected());
    assert(act.connected);
    assert(!activity);

    // 2. Intra-frame fast down+up tap in nav mode (Enter and Down arrow).
    RawKeyboardSample fast_enter[2] = { make_kbd(true, { 0x28 }), make_kbd(true, {}) };
    KeyboardPortSamples b_enter{ 0, fast_enter };
    kt.update({ &b_enter, 1 }, 1'016'000, false, &act, &txt, &activity);
    assert(activity);
    assert(act.is_pressed(hui::Action::confirm));
    assert(!act.is_held(hui::Action::confirm));

    RawKeyboardSample fast_down[2] = { make_kbd(true, { 0x51 }), make_kbd(true, {}) };
    KeyboardPortSamples b_down{ 0, fast_down };
    kt.update({ &b_down, 1 }, 1'032'000, false, &act, &txt, &activity);
    assert(activity);
    assert(act.nav == hui::Direction::down && !act.nav_repeat);

    // 3. Intra-frame fast typing burst ("ab" down/up within a single frame).
    RawKeyboardSample burst[4] = {
        make_kbd(true, { 0x04 }),
        make_kbd(true, {}),
        make_kbd(true, { 0x05 }),
        make_kbd(true, {}),
    };
    KeyboardPortSamples b_burst{ 0, burst };
    kt.update({ &b_burst, 1 }, 1'100'000, true, &act, &txt, &activity);
    assert(activity);
    assert(txt.typed == "ab");

    // 4. Multi-port independence: Port 0 idle (e.g., mouse virtual HID keyboard) while Port 1 types & holds.
    RawKeyboardSample p0_idle = make_kbd(true, {});
    RawKeyboardSample p1_c    = make_kbd(true, { 0x06 });  // 'c'
    KeyboardPortSamples both_ports[2] = {
        { 0, std::span<const RawKeyboardSample>(&p0_idle, 1) },
        { 1, std::span<const RawKeyboardSample>(&p1_c, 1) },
    };
    kt.update(both_ports, 1'200'000, true, &act, &txt, &activity);
    assert(activity);
    assert(txt.typed == "c");

    // Port 0 emits another idle sample -> must NOT release Port 1's held 'c', so 'c' still repeats at +410ms.
    feed_one(&kt, 0, p0_idle, 1'610'000, true, &act, &txt, &activity);
    assert(txt.typed == "c");
    assert(activity);

    // 5. HID rollover error (keycodes[0] == 1) on Port 1 must NOT release held 'c'.
    RawKeyboardSample rollover = make_kbd(true, { 0x01 });
    feed_one(&kt, 1, rollover, 1'670'000, true, &act, &txt, &activity);
    assert(txt.typed == "c");  // repeat still active because 'c' remained held!

    // Release 'c' on Port 1.
    RawKeyboardSample p1_idle = make_kbd(true, {});
    feed_one(&kt, 1, p1_idle, 1'700'000, true, &act, &txt, &activity);
    assert(activity);
    assert(txt.empty());

    // 6. Software CapsLock toggle when hardware locks==0, then hardware locks taking over once nonzero.
    RawKeyboardSample caps_down = make_kbd(true, { 0x39 });
    feed_one(&kt, 1, caps_down, 1'750'000, true, &act, &txt);
    feed_one(&kt, 1, p1_idle, 1'766'000, true, &act, &txt);

    RawKeyboardSample a_soft_caps = make_kbd(true, { 0x04 }, 0, 0);
    feed_one(&kt, 1, a_soft_caps, 1'782'000, true, &act, &txt);
    assert(txt.typed == "A");
    feed_one(&kt, 1, p1_idle, 1'798'000, true, &act, &txt);

    // Hardware locks becomes nonzero (NumLock=1, CapsLock=0) -> hardware locks overrides soft CapsLock -> lowercase 'a'.
    RawKeyboardSample a_hw_locks = make_kbd(true, { 0x04 }, 0, ps5lm::keyboard_bits::kNumLock);
    feed_one(&kt, 1, a_hw_locks, 1'814'000, true, &act, &txt);
    assert(txt.typed == "a");
    feed_one(&kt, 1, p1_idle, 1'830'000, true, &act, &txt);

    // 7. Disconnecting Port 0 keeps Port 1 connected; clearing Port 1 triggers disconnect & focus_lost.
    RawKeyboardSample p0_disc = make_kbd(false, {});
    feed_one(&kt, 0, p0_disc, 1'900'000, false, &act, &txt);
    assert(kt.connected());
    assert(act.connected);
    assert(!act.focus_lost);

    kt.clear_port(1);
    kt.update({}, 1'916'000, false, &act, &txt, &activity);
    assert(!kt.connected());
    assert(!act.connected);
    assert(act.focus_lost);
}

void write_fw_record_96(std::uint8_t * dst, std::uint64_t ts_us, std::uint8_t intercepted, bool connected,
                        std::uint32_t count, std::uint32_t locks, std::uint32_t modifiers,
                        std::initializer_list<std::uint16_t> keys) {
    std::memset(dst, 0, 96);
    std::memcpy(dst + 0x00, &ts_us, sizeof(ts_us));
    dst[0x08] = intercepted;
    dst[0x10] = connected ? 1u : 0u;
    std::memcpy(dst + 0x14, &count, sizeof(count));
    std::memcpy(dst + 0x18, &locks, sizeof(locks));
    std::memcpy(dst + 0x1C, &modifiers, sizeof(modifiers));
    std::size_t idx = 0;
    for (std::uint16_t k : keys) {
        if (idx < 16) {
            std::memcpy(dst + 0x20 + idx * 2, &k, sizeof(k));
            ++idx;
        }
    }
}

void test_keyboard_96byte_stride_and_modifiers() {
    // Regression: PS5 libSceKeyboard.sprx writes 96-byte (0x60) records per
    // sample. Reading a 2-sample batch with a 64-byte struct caused sample[1]
    // to read record 0's zero-filled reserve2[32] (connected=0, count=0) and
    // drop record 1 at offset 0x60.
    assert(sizeof(RawKeyboardSample) == 96);

    std::uint8_t fw_wire[16 * 96]{};
    // Record 0 (0x00): idle connected snapshot; Record 1 (0x60): Tab pressed.
    write_fw_record_96(fw_wire + 0 * 96, 1000, 0, true, 1, 0, 0, {});
    write_fw_record_96(fw_wire + 1 * 96, 2000, 0, true, 1, 0, 0, { 0x2B });

    RawKeyboardSample samples[16]{};
    std::memcpy(samples, fw_wire, 2 * sizeof(RawKeyboardSample));

    KeyboardTracker kt;
    kt.set_port_connected(1, true);
    hui::InputFrame     act;
    TextInput           txt;
    bool                activity = false;
    KeyboardPortSamples batch{ 1, std::span<const RawKeyboardSample>(samples, 2) };
    kt.update(std::span<const KeyboardPortSamples>(&batch, 1), 1'000'000, false, &act, &txt, &activity);
    assert(kt.connected());
    assert(act.connected);
    assert(!act.focus_lost);
    assert(activity);
    assert(act.is_pressed(hui::Action::page_next));

    // Full 16-sample firmware batch: record 15 at byte 15*96 must decode cleanly.
    for (int i = 0; i < 15; ++i) {
        write_fw_record_96(fw_wire + i * 96, static_cast<std::uint64_t>(3000 + i * 100), 0, true, 1, 0, 0, {});
    }
    write_fw_record_96(fw_wire + 15 * 96, 5000, 0, true, 1, 0, 0, { 0x28 });  // Enter
    std::memcpy(samples, fw_wire, 16 * sizeof(RawKeyboardSample));
    KeyboardPortSamples full_batch{ 1, std::span<const RawKeyboardSample>(samples, 16) };
    kt.update(std::span<const KeyboardPortSamples>(&full_batch, 1), 1'016'000, false, &act, &txt, &activity);
    assert(kt.connected());
    assert(act.connected);
    assert(!act.focus_lost);
    assert(act.is_pressed(hui::Action::confirm));

    // Non-slot-0 keycode (keycodes[3] = 'a', keycodes[0..2] = 0) still types in text mode.
    RawKeyboardSample idle = make_kbd(true, {});
    feed_one(&kt, 1, idle, 1'032'000, true, &act, &txt, &activity);
    RawKeyboardSample slot3{};
    slot3.connected   = 1;
    slot3.count       = 1;
    slot3.keycodes[3] = 0x04;  // 'a'
    feed_one(&kt, 1, slot3, 1'048'000, true, &act, &txt, &activity);
    assert(activity);
    assert(txt.typed == "a");

    // Modifier-only press/release (Left Shift) counts as keyboard activity without firing nav/text.
    feed_one(&kt, 1, idle, 1'064'000, false, &act, &txt, &activity);
    RawKeyboardSample shift_only = make_kbd(true, {}, ps5lm::keyboard_bits::kLeftShift);
    feed_one(&kt, 1, shift_only, 1'080'000, false, &act, &txt, &activity);
    assert(activity);
    assert(act.pressed == 0);
    assert(txt.empty());

    // Steady 0-sample frame while Shift remains held -> no activity.
    kt.update({}, 1'096'000, false, &act, &txt, &activity);
    assert(!activity);

    // Releasing Shift -> activity true.
    feed_one(&kt, 1, idle, 1'112'000, false, &act, &txt, &activity);
    assert(activity);
    assert(act.pressed == 0);

    // System overlay interception (intercepted=1) must release held keys without disconnecting.
    RawKeyboardSample down_held = make_kbd(true, { 0x51 });  // Down arrow
    feed_one(&kt, 1, down_held, 1'130'000, false, &act, &txt, &activity);
    assert(act.is_held(hui::Action::down));

    RawKeyboardSample intr = make_kbd(true, { 0x51 });
    intr.intercepted       = 1;
    feed_one(&kt, 1, intr, 1'146'000, false, &act, &txt, &activity);
    assert(kt.connected());
    assert(act.connected);
    assert(!act.is_held(hui::Action::down));
    assert((act.released & hui::action_bit(hui::Action::down)) != 0);

    // Once interception clears, normal key presses resume cleanly.
    feed_one(&kt, 1, down_held, 1'162'000, false, &act, &txt, &activity);
    assert(act.is_pressed(hui::Action::down));
    assert(act.is_held(hui::Action::down));
}

void test_mouse_tracker() {
    MouseTracker    mt;
    hui::InputFrame act;
    PointerInput    ptr;

    assert(!mt.visible());

    // Move right+down with fast intra-frame left click (down packet + up packet in same poll).
    RawMouseData pkts[2] = {
        { 100, 1, ps5lm::mouse_bits::kLeft, 100, -50, 2, 0, 0 },
        { 200, 1, 0, 10, 0, 0, 0, 0 },
    };
    mt.update(pkts, 1'000'000, false, &act, &ptr);
    assert(act.connected);
    assert(ptr.visible);
    assert(ptr.moved);
    assert(ptr.left_pressed && ptr.left_released && !ptr.left_down);
    assert(ptr.wheel == 2);
    assert(std::fabs(ptr.x - (960.0f + 110.0f * MouseTracker::kSensitivity)) < 0.01f);
    assert(std::fabs(ptr.y - (540.0f - 50.0f * MouseTracker::kSensitivity)) < 0.01f);

    // Clamping at bottom-right.
    RawMouseData big{ 300, 1, 0, 5000, 5000, 0, 0, 0 };
    mt.update({ &big, 1 }, 1'100'000, false, &act, &ptr);
    assert(ptr.x == 1919.0f && ptr.y == 1079.0f);

    // Right click -> Action::back, Back/Fwd side buttons -> page_prev/page_next.
    RawMouseData rclick{ 400, 1, ps5lm::mouse_bits::kRight | ps5lm::mouse_bits::kBack, 0, 0, 0, 0, 0 };
    mt.update({ &rclick, 1 }, 1'200'000, false, &act, &ptr);
    assert(ptr.right_pressed);
    assert(act.is_pressed(hui::Action::back));
    assert(act.is_pressed(hui::Action::page_prev));

    // Intercepted releases all held buttons.
    RawMouseData inter{ 500, 1, ps5lm::mouse_bits::kIntercepted, 0, 0, 0, 0, 0 };
    mt.update({ &inter, 1 }, 1'300'000, false, &act, &ptr);
    assert(act.focus_lost);
    assert(act.held == 0 && !ptr.left_down);

    // Auto-hide after 5s idle or immediate hide on DualSense/keyboard activity.
    RawMouseData wake{ 600, 1, 0, 4, 0, 0, 0, 0 };
    mt.update({ &wake, 1 }, 2'000'000, false, &act, &ptr);
    assert(ptr.visible);
    mt.update({}, 7'000'000, false, &act, &ptr);
    assert(!ptr.visible);
    mt.update({ &wake, 1 }, 8'000'000, false, &act, &ptr);
    assert(ptr.visible);
    mt.update({}, 8'016'000, true, &act, &ptr);
    assert(!ptr.visible);
}

void test_mouse_multiport() {
    MouseTracker    mt;
    hui::InputFrame act;
    PointerInput    ptr;

    // (a) Active port 0 unaffected by connected=0 packets on idle/unattached port 1.
    RawMouseData     p0_hold{ 100, 1, ps5lm::mouse_bits::kLeft | ps5lm::mouse_bits::kRight, 20, 10, 0, 0, 0 };
    MousePortSamples b0{ 0, std::span<const RawMouseData>(&p0_hold, 1) };
    mt.update(std::span<const MousePortSamples>(&b0, 1), 1'000'000, false, &act, &ptr);
    assert(mt.connected());
    assert(act.connected);
    assert(!act.focus_lost);
    assert(ptr.left_down && ptr.left_pressed && ptr.right_pressed);
    assert(act.is_held(hui::Action::back));
    const float saved_x = ptr.x;
    const float saved_y = ptr.y;

    RawMouseData     p1_unattached{ 200, 0, ps5lm::mouse_bits::kMiddle, 40, 40, 2, 0, 0 };
    MousePortSamples b1_idle{ 1, std::span<const RawMouseData>(&p1_unattached, 1) };
    mt.update(std::span<const MousePortSamples>(&b1_idle, 1), 1'016'000, false, &act, &ptr);
    assert(mt.connected());
    assert(act.connected);
    assert(!act.focus_lost);
    assert(ptr.left_down && !ptr.left_pressed && !ptr.left_released);
    assert(act.is_held(hui::Action::back));
    assert(act.released == 0);
    assert(!ptr.moved && ptr.wheel == 0);
    assert(std::fabs(ptr.x - saved_x) < 0.01f && std::fabs(ptr.y - saved_y) < 0.01f);

    RawMouseData     p0_move{ 250, 1, ps5lm::mouse_bits::kLeft | ps5lm::mouse_bits::kRight, 4, 0, 0, 0, 0 };
    MousePortSamples mixed[2] = {
        { 0, std::span<const RawMouseData>(&p0_move, 1) },
        { 1, std::span<const RawMouseData>(&p1_unattached, 1) },
    };
    mt.update(mixed, 1'032'000, false, &act, &ptr);
    assert(mt.connected() && act.connected && !act.focus_lost);
    assert(ptr.left_down && !ptr.left_released && act.is_held(hui::Action::back));
    assert(ptr.moved && std::fabs(ptr.x - (saved_x + 4.0f * MouseTracker::kSensitivity)) < 0.01f);

    // (b) Simultaneous 2-port activity + disconnect of port 1 leaves port 0 intact without focus_lost.
    RawMouseData     p0_left{ 300, 1, ps5lm::mouse_bits::kLeft, 0, 0, 0, 0, 0 };
    RawMouseData     p1_back{ 300, 1, ps5lm::mouse_bits::kBack, 10, 0, 0, 0, 0 };
    MousePortSamples both[2] = {
        { 0, std::span<const RawMouseData>(&p0_left, 1) },
        { 1, std::span<const RawMouseData>(&p1_back, 1) },
    };
    mt.update(both, 1'048'000, false, &act, &ptr);
    assert(mt.connected() && act.connected && !act.focus_lost);
    assert(ptr.left_down && !ptr.left_released);
    assert(act.is_pressed(hui::Action::page_prev) && act.is_held(hui::Action::page_prev));
    assert((act.released & hui::action_bit(hui::Action::back)) != 0);

    RawMouseData     p1_disc{ 400, 0, 0, 0, 0, 0, 0, 0 };
    MousePortSamples b1_disc{ 1, std::span<const RawMouseData>(&p1_disc, 1) };
    mt.update(std::span<const MousePortSamples>(&b1_disc, 1), 1'100'000, false, &act, &ptr);
    assert(mt.connected() && act.connected && !act.focus_lost);
    assert(ptr.left_down && !ptr.left_released);
    assert(!act.is_held(hui::Action::page_prev));
    assert((act.released & hui::action_bit(hui::Action::page_prev)) != 0);

    // (c) kIntercepted on active port sets focus_lost and releases held buttons cleanly.
    RawMouseData     p0_hold_both{ 450, 1, ps5lm::mouse_bits::kLeft | ps5lm::mouse_bits::kForward, 0, 0, 0, 0, 0 };
    RawMouseData     p1_ready{ 450, 1, 0, 0, 0, 0, 0, 0 };
    MousePortSamples two_active[2] = {
        { 0, std::span<const RawMouseData>(&p0_hold_both, 1) },
        { 1, std::span<const RawMouseData>(&p1_ready, 1) },
    };
    mt.update(two_active, 1'150'000, false, &act, &ptr);
    assert(ptr.left_down && act.is_held(hui::Action::page_next));

    RawMouseData     p0_inter{ 500, 1, ps5lm::mouse_bits::kIntercepted, 0, 0, 0, 0, 0 };
    MousePortSamples b0_inter{ 0, std::span<const RawMouseData>(&p0_inter, 1) };
    mt.update(std::span<const MousePortSamples>(&b0_inter, 1), 1'200'000, false, &act, &ptr);
    assert(act.focus_lost);
    assert(!ptr.left_down && ptr.left_released);
    assert(!act.is_held(hui::Action::page_next));
    assert((act.released & hui::action_bit(hui::Action::page_next)) != 0);
    assert(mt.connected() && act.connected);  // port 1 still connected

    mt.update(std::span<const MousePortSamples>(&b1_disc, 1), 1'250'000, false, &act, &ptr);
    assert(!mt.connected() && !act.connected && act.focus_lost);

    // (d) clear_port(0) on last active port followed by 0-sample frame releases held buttons and sets focus_lost.
    mt.update(std::span<const MousePortSamples>(&b0, 1), 1'300'000, false, &act, &ptr);
    assert(mt.connected() && ptr.left_down && act.is_held(hui::Action::back));
    mt.clear_port(0);
    mt.update({}, 1'350'000, false, &act, &ptr);
    assert(!mt.connected() && !act.connected && act.focus_lost);
    assert(ptr.left_released && !ptr.left_down);
    assert(act.held == 0 && (act.released & hui::action_bit(hui::Action::back)) != 0);

    // (e) Out-of-range ports (-1, kMaxPorts) are safely ignored.
    mt.update(std::span<const MousePortSamples>(&b0, 1), 1'400'000, false, &act, &ptr);
    const float oob_x = ptr.x;
    const float oob_y = ptr.y;
    mt.clear_port(-1);
    mt.clear_port(MouseTracker::kMaxPorts);

    RawMouseData     oob_pkt{ 600, 1, ps5lm::mouse_bits::kForward, 50, 50, 3, 0, 0 };
    RawMouseData     oob_disc{ 601, 0, ps5lm::mouse_bits::kIntercepted, 0, 0, 0, 0, 0 };
    MousePortSamples oob_batches[4] = {
        { -1, std::span<const RawMouseData>(&oob_pkt, 1) },
        { MouseTracker::kMaxPorts, std::span<const RawMouseData>(&oob_pkt, 1) },
        { -1, std::span<const RawMouseData>(&oob_disc, 1) },
        { MouseTracker::kMaxPorts, std::span<const RawMouseData>(&oob_disc, 1) },
    };
    mt.update(oob_batches, 1'450'000, false, &act, &ptr);
    assert(mt.connected() && act.connected && !act.focus_lost);
    assert(ptr.left_down && !ptr.left_released);
    assert(act.is_held(hui::Action::page_next) == false && act.is_held(hui::Action::back));
    assert(act.pressed == 0 && act.released == 0);
    assert(!ptr.moved && ptr.wheel == 0);
    assert(std::fabs(ptr.x - oob_x) < 0.01f && std::fabs(ptr.y - oob_y) < 0.01f);
}

void test_remote_and_merge() {
    RemoteTracker   rt;
    RawRemoteSample rs{ hui::pad_bits::kUp, true, 13 };  // D-pad up + CEC Enter
    hui::InputFrame rf = rt.update(rs, 1'000'000);
    assert(rf.connected);
    assert(rf.nav == hui::Direction::up);
    assert(rf.is_pressed(hui::Action::confirm));

    hui::InputFrame disconnected_pad;
    disconnected_pad.connected  = false;
    disconnected_pad.focus_lost = true;

    const hui::InputFrame merged = merge_input(disconnected_pad, rf);
    assert(merged.connected);
    assert(merged.focus_lost);
    assert(merged.nav == hui::Direction::up);
    assert(merged.is_pressed(hui::Action::confirm));

    hui::InputFrame active_pad;
    active_pad.connected  = true;
    active_pad.focus_lost = false;
    hui::InputFrame lost_usb;
    lost_usb.connected  = false;
    lost_usb.focus_lost = true;
    assert(merge_input(active_pad, lost_usb).focus_lost);

    hui::InputFrame steady_usb;
    steady_usb.connected  = true;
    steady_usb.focus_lost = false;
    assert(!merge_input(active_pad, steady_usb).focus_lost);
}

}  // namespace

int main() {
    test_hid_decoding();
    test_keyboard_nav_and_rollover();
    test_keyboard_text_mode_and_transition_guard();
    test_keyboard_fast_taps_multiport_errors_and_capslock();
    test_keyboard_96byte_stride_and_modifiers();
    test_mouse_tracker();
    test_mouse_multiport();
    test_remote_and_merge();
    std::printf("usb_input_test: ok\n");
    return 0;
}
