// PS5 USB keyboard, mouse, and HDMI-CEC media remote input alongside DualSense.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "usb_input.hpp"
#include "log.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace ps5lm {

namespace {

constexpr std::uint32_t kRepeatDelayUs        = 350'000;
constexpr std::uint32_t kRepeatIntervalUs     = 110'000;
constexpr std::uint32_t kTextRepeatDelayUs    = 400'000;
constexpr std::uint32_t kTextRepeatIntervalUs = 55'000;

constexpr std::uint16_t kHidMinValid         = 0x04;
constexpr std::uint16_t kHidMaxValid         = 0xE7;
constexpr int           kMaxKeyboardKeySlots = 16;

constexpr std::uint16_t kKeyReturn      = 0x28;
constexpr std::uint16_t kKeyEscape      = 0x29;
constexpr std::uint16_t kKeyBackspace   = 0x2A;
constexpr std::uint16_t kKeyTab         = 0x2B;
constexpr std::uint16_t kKeySpace       = 0x2C;
constexpr std::uint16_t kKeyMinus       = 0x2D;
constexpr std::uint16_t kKeyEquals      = 0x2E;
constexpr std::uint16_t kKeyLBracket    = 0x2F;
constexpr std::uint16_t kKeyRBracket    = 0x30;
constexpr std::uint16_t kKeyComma       = 0x36;
constexpr std::uint16_t kKeyPeriod      = 0x37;
constexpr std::uint16_t kKeySlash       = 0x38;
constexpr std::uint16_t kKeyCapsLock    = 0x39;
constexpr std::uint16_t kKeyF1          = 0x3A;
constexpr std::uint16_t kKeyF2          = 0x3B;
constexpr std::uint16_t kKeyF10         = 0x43;
constexpr std::uint16_t kKeyHome        = 0x4A;
constexpr std::uint16_t kKeyPageUp      = 0x4B;
constexpr std::uint16_t kKeyDelete      = 0x4C;
constexpr std::uint16_t kKeyEnd         = 0x4D;
constexpr std::uint16_t kKeyPageDown    = 0x4E;
constexpr std::uint16_t kKeyRight       = 0x4F;
constexpr std::uint16_t kKeyLeft        = 0x50;
constexpr std::uint16_t kKeyDown        = 0x51;
constexpr std::uint16_t kKeyUp          = 0x52;
constexpr std::uint16_t kKeyKpEnter     = 0x58;
constexpr std::uint16_t kKeyApplication = 0x65;
constexpr std::uint16_t kKeyE           = 0x08;
constexpr std::uint16_t kKeyQ           = 0x14;
constexpr std::uint16_t kKeyT           = 0x17;
constexpr std::uint16_t kKeyX           = 0x1B;
constexpr std::uint16_t kKeyY           = 0x1C;
constexpr std::uint16_t kKeyLeftCtrl    = 0xE0;
constexpr std::uint16_t kKeyRightGui    = 0xE7;

hui::Direction direction_from_actions(std::uint32_t actions) {
    if ((actions & hui::action_bit(hui::Action::up)) != 0) {
        return hui::Direction::up;
    }
    if ((actions & hui::action_bit(hui::Action::down)) != 0) {
        return hui::Direction::down;
    }
    if ((actions & hui::action_bit(hui::Action::left)) != 0) {
        return hui::Direction::left;
    }
    if ((actions & hui::action_bit(hui::Action::right)) != 0) {
        return hui::Direction::right;
    }
    return hui::Direction::none;
}

void update_nav_repeat(std::uint32_t pressed, std::uint32_t held, std::uint64_t now_us,
                       hui::Direction * nav_held, std::uint64_t * nav_next_us, hui::InputFrame * frame) {
    const hui::Direction current = direction_from_actions(held);
    const hui::Direction tapped  = direction_from_actions(pressed);
    if (tapped != hui::Direction::none && tapped != current) {
        frame->nav = tapped;
    } else if (current != *nav_held && current != hui::Direction::none) {
        frame->nav   = current;
        *nav_next_us = now_us + kRepeatDelayUs;
    } else if (current != hui::Direction::none && now_us >= *nav_next_us) {
        frame->nav        = current;
        frame->nav_repeat = true;
        *nav_next_us      = now_us + kRepeatIntervalUs;
    }
    *nav_held = current;
}

std::uint32_t map_nav_actions(const std::array<bool, 256> & down, std::uint32_t modifiers) {
    std::uint32_t actions = 0;
    const auto    set     = [&](hui::Action a) { actions |= hui::action_bit(a); };
    const bool    shift   = (modifiers & (keyboard_bits::kLeftShift | keyboard_bits::kRightShift)) != 0;

    if (down[kKeyUp]) {
        set(hui::Action::up);
    }
    if (down[kKeyDown]) {
        set(hui::Action::down);
    }
    if (down[kKeyLeft]) {
        set(hui::Action::left);
    }
    if (down[kKeyRight]) {
        set(hui::Action::right);
    }
    if (down[kKeyReturn] || down[kKeyKpEnter] || down[kKeySpace]) {
        set(hui::Action::confirm);
    }
    if (down[kKeyEscape] || down[kKeyBackspace]) {
        set(hui::Action::back);
    }
    if (down[kKeyTab]) {
        set(shift ? hui::Action::page_prev : hui::Action::page_next);
    }
    if (down[kKeyLBracket] || down[kKeyPageUp]) {
        set(hui::Action::page_prev);
    }
    if (down[kKeyRBracket] || down[kKeyPageDown]) {
        set(hui::Action::page_next);
    }
    if (down[kKeyComma] || down[kKeyMinus] || down[kKeyQ] || down[kKeyHome]) {
        set(hui::Action::jump_prev);
    }
    if (down[kKeyPeriod] || down[kKeyEquals] || down[kKeyE] || down[kKeyEnd]) {
        set(hui::Action::jump_next);
    }
    if (down[kKeyX] || down[kKeyF1] || down[kKeyDelete]) {
        set(hui::Action::west);
    }
    if (down[kKeyT] || down[kKeyY] || down[kKeyF2] || down[kKeySlash]) {
        set(hui::Action::north);
    }
    if (down[kKeyF10] || down[kKeyApplication]) {
        set(hui::Action::menu);
    }
    return actions;
}

std::uint32_t map_remote_actions(const RawRemoteSample & sample) {
    if (!sample.connected) {
        return 0;
    }
    std::uint32_t actions = 0;
    const auto    set     = [&](hui::Action a) { actions |= hui::action_bit(a); };
    const std::uint32_t b = sample.buttons;

    if ((b & hui::pad_bits::kUp) != 0) {
        set(hui::Action::up);
    }
    if ((b & hui::pad_bits::kDown) != 0) {
        set(hui::Action::down);
    }
    if ((b & hui::pad_bits::kLeft) != 0) {
        set(hui::Action::left);
    }
    if ((b & hui::pad_bits::kRight) != 0) {
        set(hui::Action::right);
    }
    if ((b & hui::pad_bits::kCross) != 0) {
        set(hui::Action::confirm);
    }
    if ((b & hui::pad_bits::kCircle) != 0) {
        set(hui::Action::back);
    }
    if ((b & hui::pad_bits::kOptions) != 0) {
        set(hui::Action::menu);
    }

    switch (sample.cec_code) {
        case 13: set(hui::Action::confirm); break;    // Enter / OK
        case 15:
        case 43: set(hui::Action::back); break;       // Back / Prev
        case 18: set(hui::Action::menu); break;       // Menu
        case 41: set(hui::Action::page_prev); break;  // Ch+ / Page-
        case 42: set(hui::Action::page_next); break;  // Ch- / Page+
        case 20: set(hui::Action::jump_prev); break;  // Prev Track
        case 21: set(hui::Action::jump_next); break;  // Next Track
        case 36:                                      // Blue
        case 37: set(hui::Action::west); break;       // Red
        case 38:                                      // Green
        case 39: set(hui::Action::north); break;      // Yellow
        default: break;
    }
    return actions;
}

std::uint32_t map_mouse_actions(std::uint32_t buttons) {
    std::uint32_t actions = 0;
    if ((buttons & mouse_bits::kRight) != 0) {
        actions |= hui::action_bit(hui::Action::back);
    }
    if ((buttons & mouse_bits::kBack) != 0) {
        actions |= hui::action_bit(hui::Action::page_prev);
    }
    if ((buttons & mouse_bits::kForward) != 0) {
        actions |= hui::action_bit(hui::Action::page_next);
    }
    return actions;
}

}  // namespace

char hid_scancode_to_ascii(std::uint16_t scancode, std::uint32_t modifiers, std::uint32_t locks) {
    if ((modifiers & (keyboard_bits::kLeftCtrl | keyboard_bits::kRightCtrl | keyboard_bits::kLeftAlt |
                      keyboard_bits::kRightAlt | keyboard_bits::kLeftGui | keyboard_bits::kRightGui)) != 0) {
        return '\0';
    }
    const bool shift = (modifiers & (keyboard_bits::kLeftShift | keyboard_bits::kRightShift)) != 0;
    const bool caps  = (locks & keyboard_bits::kCapsLock) != 0;

    if (scancode >= 0x04 && scancode <= 0x1D) {
        const char base = static_cast<char>('a' + (scancode - 0x04));
        return (shift ^ caps) ? static_cast<char>(base - 'a' + 'A') : base;
    }
    if (scancode >= 0x1E && scancode <= 0x27) {
        static constexpr char kDigits[]  = "1234567890";
        static constexpr char kShifted[] = "!@#$%^&*()";
        const int             idx        = static_cast<int>(scancode - 0x1E);
        return shift ? kShifted[idx] : kDigits[idx];
    }
    if (scancode == kKeySpace) {
        return ' ';
    }
    if (scancode >= 0x2D && scancode <= 0x38) {
        static constexpr char kPlain[]   = "-=[]\\#;'`,./";
        static constexpr char kShifted[] = "_+{}|~:\"~<>?";
        const int             idx        = static_cast<int>(scancode - 0x2D);
        return shift ? kShifted[idx] : kPlain[idx];
    }
    if (scancode == 0x64) {
        return shift ? '|' : '\\';
    }
    if (scancode == 0x54) {
        return '/';
    }
    if (scancode == 0x55) {
        return '*';
    }
    if (scancode == 0x56) {
        return '-';
    }
    if (scancode == 0x57) {
        return '+';
    }
    if (scancode >= 0x59 && scancode <= 0x61) {
        return static_cast<char>('1' + (scancode - 0x59));
    }
    if (scancode == 0x62) {
        return '0';
    }
    if (scancode == 0x63) {
        return '.';
    }
    return '\0';
}

void KeyboardTracker::set_port_connected(int port, bool connected) {
    if (port < 0 || port >= kMaxPorts) {
        return;
    }
    PortState & ps = ports_[static_cast<std::size_t>(port)];
    ps.connected   = connected;
    if (!connected) {
        ps.down.fill(false);
        ps.modifiers = 0;
        ps.locks     = 0;
    }
    bool any = false;
    for (const PortState & p : ports_) {
        if (p.connected) {
            any = true;
            break;
        }
    }
    connected_ = any;
}

void KeyboardTracker::clear_port(int port) {
    if (port < 0 || port >= kMaxPorts) {
        return;
    }
    ports_[static_cast<std::size_t>(port)] = {};
}

void KeyboardTracker::update(std::span<const KeyboardPortSamples> port_batches, std::uint64_t now_us, bool text_mode,
                             hui::InputFrame * out_actions, TextInput * out_text, bool * out_activity) {
    hui::InputFrame       frame;
    TextInput             text;
    std::array<bool, 256> tapped_down{};
    std::uint32_t         tapped_mods = 0;
    bool                  activity    = false;

    if (text_mode && !prev_text_mode_) {
        repeat_code_      = 0;
        repeat_backspace_ = false;
        repeat_next_us_   = 0;
    }

    const auto recompute_aggregate = [&](std::array<bool, 256> * out_down, std::uint32_t * out_mods,
                                         std::uint32_t * out_locks, bool * out_conn) {
        out_down->fill(false);
        std::uint32_t mods = 0;
        std::uint32_t lks  = 0;
        bool          conn = false;
        for (const PortState & p : ports_) {
            if (!p.connected) {
                continue;
            }
            conn = true;
            mods |= p.modifiers;
            lks |= p.locks;
            for (std::uint16_t c = kHidMinValid; c <= kHidMaxValid; ++c) {
                if (p.down[c]) {
                    (*out_down)[c] = true;
                }
            }
        }
        *out_mods = mods;
        *out_locks = lks;
        *out_conn = conn;
    };

    const auto effective_locks = [&](std::uint32_t hw_locks) -> std::uint32_t {
        if (locks_seen_) {
            return hw_locks;
        }
        return soft_caps_lock_ ? keyboard_bits::kCapsLock : 0u;
    };

    for (const KeyboardPortSamples & batch : port_batches) {
        if (batch.port < 0 || batch.port >= kMaxPorts) {
            continue;
        }
        PortState & ps = ports_[static_cast<std::size_t>(batch.port)];
        for (const RawKeyboardSample & s : batch.samples) {
            if (s.connected != 0 && s.intercepted == 0 && s.keycodes[0] >= 1 && s.keycodes[0] <= 3) {
                // HID rollover/POST/undefined error: keep existing held state.
                ps.connected = true;
                continue;
            }

            const bool user_event = (s.connected != 0 && s.intercepted == 0);
            ps.connected          = (s.connected != 0);
            if (!user_event) {
                ps.down.fill(false);
                ps.modifiers = 0;
                ps.locks     = 0;
            } else {
                ps.down.fill(false);
                std::uint32_t pmods  = s.modifiers & 0xFFu;
                std::uint32_t plocks = s.locks & 0x07u;
                if (plocks != 0) {
                    locks_seen_ = true;
                }
                for (int i = 0; i < kMaxKeyboardKeySlots; ++i) {
                    const std::uint16_t code = s.keycodes[i];
                    if (code >= kHidMinValid && code <= kHidMaxValid) {
                        ps.down[code] = true;
                        if (code >= kKeyLeftCtrl && code <= kKeyRightGui) {
                            pmods |= (1u << (code - kKeyLeftCtrl));
                        }
                    }
                }
                ps.modifiers = pmods;
                ps.locks     = plocks;
            }

            std::array<bool, 256> agg_down{};
            std::uint32_t         agg_mods  = 0;
            std::uint32_t         agg_locks = 0;
            bool                  agg_conn  = false;
            recompute_aggregate(&agg_down, &agg_mods, &agg_locks, &agg_conn);
            if (user_event && agg_conn && agg_mods != prev_modifiers_) {
                activity = true;
            }
            prev_modifiers_ = agg_conn ? agg_mods : 0u;

            for (std::uint16_t code = kHidMinValid; code <= kHidMaxValid; ++code) {
                if (agg_down[code] && !down_[code]) {
                    activity          = true;
                    tapped_down[code] = true;
                    tapped_mods |= agg_mods;
                    if (code == kKeyCapsLock) {
                        soft_caps_lock_ = !soft_caps_lock_;
                    }
                    if (text_mode) {
                        if (code == kKeyEscape) {
                            text.cancel       = true;
                            repeat_code_      = 0;
                            repeat_backspace_ = false;
                        } else if (code == kKeyReturn || code == kKeyKpEnter) {
                            text.submit       = true;
                            repeat_code_      = 0;
                            repeat_backspace_ = false;
                        } else if (code == kKeyBackspace) {
                            ++text.backspaces;
                            repeat_code_      = code;
                            repeat_backspace_ = true;
                            repeat_next_us_   = now_us + kTextRepeatDelayUs;
                        } else {
                            const char ch = hid_scancode_to_ascii(code, agg_mods, effective_locks(agg_locks));
                            if (ch != '\0') {
                                text.typed.push_back(ch);
                                repeat_code_      = code;
                                repeat_backspace_ = false;
                                repeat_next_us_   = now_us + kTextRepeatDelayUs;
                            }
                        }
                    }
                } else if (!agg_down[code] && down_[code]) {
                    if (user_event) {
                        activity = true;
                    }
                    if (repeat_code_ == code) {
                        repeat_code_      = 0;
                        repeat_backspace_ = false;
                    }
                }
            }
            down_ = agg_down;
        }
    }

    std::array<bool, 256> agg_down{};
    std::uint32_t         agg_mods  = 0;
    std::uint32_t         agg_locks = 0;
    bool                  agg_conn  = false;
    recompute_aggregate(&agg_down, &agg_mods, &agg_locks, &agg_conn);
    down_ = agg_down;

    if (!agg_conn) {
        if (connected_ || held_actions_ != 0) {
            frame.focus_lost = true;
            frame.released   = held_actions_;
        }
        down_.fill(false);
        connected_        = false;
        prev_modifiers_   = 0;
        held_actions_     = 0;
        nav_held_         = hui::Direction::none;
        nav_next_us_      = 0;
        repeat_code_      = 0;
        repeat_backspace_ = false;
        repeat_next_us_   = 0;
        prev_text_mode_   = text_mode;
        *out_actions      = frame;
        *out_text         = std::move(text);
        if (out_activity != nullptr) {
            *out_activity = activity;
        }
        return;
    }

    connected_      = true;
    frame.connected = true;

    if (text_mode) {
        if (repeat_code_ != 0) {
            if (!down_[repeat_code_]) {
                repeat_code_      = 0;
                repeat_backspace_ = false;
            } else {
                int steps = 0;
                while (now_us >= repeat_next_us_ && steps < 4) {
                    if (repeat_backspace_) {
                        ++text.backspaces;
                        activity = true;
                    } else {
                        const char ch = hid_scancode_to_ascii(repeat_code_, agg_mods, effective_locks(agg_locks));
                        if (ch == '\0') {
                            repeat_code_ = 0;
                            break;
                        }
                        text.typed.push_back(ch);
                        activity = true;
                    }
                    repeat_next_us_ += kTextRepeatIntervalUs;
                    ++steps;
                }
                if (now_us > repeat_next_us_) {
                    repeat_next_us_ = now_us + kTextRepeatIntervalUs;
                }
            }
        }

        // Keep arrow navigation active in text mode so the user can move onto
        // Turkish or symbol keys on the on-screen keyboard too.
        std::array<bool, 256> arrows{};
        std::array<bool, 256> tapped_arrows{};
        for (std::uint16_t ac : { kKeyRight, kKeyLeft, kKeyDown, kKeyUp }) {
            arrows[ac]        = down_[ac];
            tapped_arrows[ac] = tapped_down[ac];
        }
        const std::uint32_t held_now    = map_nav_actions(arrows, agg_mods);
        const std::uint32_t tapped_acts = map_nav_actions(tapped_arrows, agg_mods);
        const std::uint32_t press_mask  = held_now | tapped_acts;
        frame.pressed |= press_mask & ~held_actions_;
        frame.released |= (held_actions_ & ~held_now) | (tapped_acts & ~held_now & ~held_actions_);
        held_actions_ = held_now;
        frame.held    = held_actions_;
        update_nav_repeat(frame.pressed, frame.held, now_us, &nav_held_, &nav_next_us_, &frame);
    } else {
        repeat_code_      = 0;
        repeat_backspace_ = false;
        repeat_next_us_   = 0;

        const std::uint32_t held_now    = map_nav_actions(down_, agg_mods);
        const std::uint32_t tapped_acts = map_nav_actions(tapped_down, agg_mods | tapped_mods);
        const std::uint32_t press_mask  = held_now | tapped_acts;
        if (prev_text_mode_) {
            // Do not re-trigger held Enter/Esc right after InputPrompt closes.
            held_actions_ = held_now;
            nav_held_     = direction_from_actions(held_actions_);
            nav_next_us_  = now_us + kRepeatDelayUs;
        } else {
            frame.pressed |= press_mask & ~held_actions_;
            frame.released |= (held_actions_ & ~held_now) | (tapped_acts & ~held_now & ~held_actions_);
            held_actions_ = held_now;
        }
        frame.held = held_actions_;
        if (!prev_text_mode_) {
            update_nav_repeat(frame.pressed, frame.held, now_us, &nav_held_, &nav_next_us_, &frame);
        }
    }

    prev_text_mode_ = text_mode;
    const bool had_activity =
        activity || frame.nav != hui::Direction::none || frame.pressed != 0 || !text.empty();
    *out_actions = frame;
    *out_text    = std::move(text);
    if (out_activity != nullptr) {
        *out_activity = had_activity;
    }
}

void MouseTracker::clear_port(int port) {
    if (port < 0 || port >= kMaxPorts) {
        return;
    }
    ports_[static_cast<std::size_t>(port)] = {};
}

void MouseTracker::update(std::span<const MousePortSamples> port_batches, std::uint64_t now_us, bool hide_cursor,
                          hui::InputFrame * out_actions, PointerInput * out_pointer) {
    hui::InputFrame frame;
    PointerInput    ptr;
    ptr.x = x_;
    ptr.y = y_;

    bool had_activity = false;
    bool lost         = false;

    const auto recompute_aggregate = [&](bool * out_conn, std::uint32_t * out_btns) {
        bool          conn = false;
        std::uint32_t btns = 0;
        for (const PortState & p : ports_) {
            if (!p.connected) {
                continue;
            }
            conn = true;
            btns |= p.buttons;
        }
        *out_conn = conn;
        *out_btns = btns;
    };

    const auto apply_aggregate = [&](bool usable_event, bool intercepted, bool was_port_active) {
        bool          agg_conn = false;
        std::uint32_t agg_btns = 0;
        recompute_aggregate(&agg_conn, &agg_btns);

        const std::uint32_t edge_down      = agg_btns & ~held_buttons_;
        const std::uint32_t edge_up        = held_buttons_ & ~agg_btns;
        const bool          was_any_active = connected_ || held_buttons_ != 0;
        if (usable_event && (edge_down != 0 || edge_up != 0)) {
            had_activity = true;
        }
        if ((edge_down & mouse_bits::kLeft) != 0) {
            ptr.left_pressed = true;
        }
        if ((edge_up & mouse_bits::kLeft) != 0) {
            ptr.left_released = true;
        }
        if ((edge_down & mouse_bits::kRight) != 0) {
            ptr.right_pressed = true;
        }
        if ((edge_down & mouse_bits::kMiddle) != 0) {
            ptr.middle_pressed = true;
        }
        held_buttons_ = agg_btns;

        const std::uint32_t actions = map_mouse_actions(agg_btns);
        frame.pressed |= actions & ~held_actions_;
        frame.released |= held_actions_ & ~actions;
        held_actions_ = actions;

        if ((intercepted && was_port_active) || (!agg_conn && was_any_active)) {
            lost = true;
        }
        connected_ = agg_conn;
    };

    for (const MousePortSamples & batch : port_batches) {
        if (batch.port < 0 || batch.port >= kMaxPorts) {
            continue;
        }
        PortState & ps = ports_[static_cast<std::size_t>(batch.port)];
        for (const RawMouseData & pkt : batch.samples) {
            const bool intercepted = (pkt.buttons & mouse_bits::kIntercepted) != 0;
            const bool usable      = pkt.connected != 0 && !intercepted;
            if (!usable) {
                const bool was_port_active = ps.connected || ps.buttons != 0;
                ps                         = {};
                apply_aggregate(false, intercepted, was_port_active);
                continue;
            }

            ps.connected = true;
            if (pkt.x_axis != 0 || pkt.y_axis != 0) {
                x_ = std::clamp(x_ + static_cast<float>(pkt.x_axis) * kSensitivity, 0.0f, 1919.0f);
                y_ = std::clamp(y_ + static_cast<float>(pkt.y_axis) * kSensitivity, 0.0f, 1079.0f);
                ptr.moved    = true;
                had_activity = true;
            }
            if (pkt.wheel != 0) {
                ptr.wheel += pkt.wheel;
                had_activity = true;
            }

            ps.buttons = pkt.buttons & 0x1Fu;
            apply_aggregate(true, false, true);
        }
    }

    // Reconcile pending clear_port() calls even on 0-sample frames.
    apply_aggregate(false, false, false);

    if (had_activity) {
        visible_          = true;
        last_activity_us_ = now_us;
    } else if (hide_cursor || (visible_ && now_us >= last_activity_us_ + kCursorHideDelayUs)) {
        visible_ = false;
    }

    ptr.x         = x_;
    ptr.y         = y_;
    ptr.visible   = visible_;
    ptr.left_down = (held_buttons_ & mouse_bits::kLeft) != 0;

    frame.held       = held_actions_;
    frame.connected  = connected_;
    frame.focus_lost = lost;

    *out_actions = frame;
    *out_pointer = ptr;
}

hui::InputFrame RemoteTracker::update(const RawRemoteSample & sample, std::uint64_t now_us) {
    hui::InputFrame frame;
    if (!sample.connected) {
        if (connected_ || held_actions_ != 0) {
            frame.focus_lost = true;
            frame.released   = held_actions_;
        }
        connected_    = false;
        held_actions_ = 0;
        nav_held_     = hui::Direction::none;
        nav_next_us_  = 0;
        return frame;
    }

    connected_                      = true;
    const std::uint32_t actions_now = map_remote_actions(sample);
    frame.pressed                   = actions_now & ~held_actions_;
    frame.released                  = held_actions_ & ~actions_now;
    held_actions_                   = actions_now;
    frame.held                      = held_actions_;
    frame.connected                 = true;
    update_nav_repeat(frame.pressed, frame.held, now_us, &nav_held_, &nav_next_us_, &frame);
    return frame;
}

hui::InputFrame merge_input(const hui::InputFrame & pad, const hui::InputFrame & usb) {
    hui::InputFrame out = pad;
    out.pressed |= usb.pressed;
    out.released |= usb.released;
    out.held |= usb.held;
    out.connected  = pad.connected || usb.connected;
    out.focus_lost = pad.focus_lost || usb.focus_lost;
    if (out.nav == hui::Direction::none && usb.nav != hui::Direction::none) {
        out.nav            = usb.nav;
        out.nav_repeat     = usb.nav_repeat;
        out.nav_from_stick = false;
    }
    return out;
}

}  // namespace ps5lm

#ifdef __PROSPERO__

namespace {

constexpr std::uint16_t kSysmoduleMouse    = 0x00A9;
constexpr std::uint16_t kSysmoduleKeyboard = 0x0106;

constexpr int kPadPortTypeRemoteControl = 16;
constexpr int kSystemUserId             = 0xFF;

constexpr std::uint32_t kErrAlreadyInitialized = 0x80960003u;
constexpr std::uint32_t kErrPadAlreadyOpened   = 0x80920004u;
constexpr std::uint32_t kErrDeviceInvalidIndex = 0x809B0003u;
constexpr std::uint32_t kErrKbdInvalidArg      = 0x80DA0001u;
constexpr std::uint32_t kErrKbdInvalidPort     = 0x80DA0002u;
constexpr std::uint32_t kErrKbdInvalidHandle   = 0x80DA0003u;
constexpr std::uint32_t kErrKbdAlreadyOpened   = 0x80DA0004u;
constexpr std::uint32_t kErrMouseInvalidArg    = 0x80DF0001u;
constexpr std::uint32_t kErrMouseInvalidHandle = 0x80DF0003u;
constexpr std::uint32_t kErrMouseAlreadyOpened = 0x80DF0004u;

struct RawPadState {
    std::uint32_t buttons;
    std::uint8_t  left_x;
    std::uint8_t  left_y;
    std::uint8_t  right_x;
    std::uint8_t  right_y;
    std::uint8_t  l2;
    std::uint8_t  r2;
    std::uint8_t  reserved0[66];
    std::int32_t  connected;
    std::uint64_t timestamp_us;
    std::uint8_t  extension[16];
    std::uint8_t  connected_count;
    std::uint8_t  reserved1[15];
};
static_assert(sizeof(RawPadState) == 120);
static_assert(offsetof(RawPadState, connected) == 0x4c);
static_assert(offsetof(RawPadState, reserved1) == 0x69);

using FnKbdInit    = int (*)(void);
using FnKbdOpen    = int (*)(int, int, int, const void *);
using FnKbdClose   = int (*)(int);
using FnKbdRead    = int (*)(int, ps5lm::RawKeyboardSample *, int);
using FnKbdGetConn = int (*)(int, std::uint8_t *);

using FnMouseInit  = int (*)(void);
using FnMouseOpen  = int (*)(int, int, int, const void *);
using FnMouseClose = int (*)(int);
using FnMouseRead  = int (*)(int, ps5lm::RawMouseData *, int);

struct ResolvedApi {
    int          module_handle = -1;
    FnKbdInit    kbd_init      = nullptr;
    FnKbdOpen    kbd_open      = nullptr;
    FnKbdClose   kbd_close     = nullptr;
    FnKbdRead    kbd_read      = nullptr;
    FnKbdGetConn kbd_get_conn  = nullptr;

    int          mouse_handle = -1;
    FnMouseInit  mouse_init   = nullptr;
    FnMouseOpen  mouse_open   = nullptr;
    FnMouseClose mouse_close  = nullptr;
    FnMouseRead  mouse_read   = nullptr;
};

ResolvedApi   g_api;
std::uint32_t g_last_kbd_open_err   = 0;
std::uint32_t g_last_mouse_open_err = 0;

extern "C" {
int          sceSysmoduleLoadModule(std::uint16_t id);
const char * sceKernelGetFsSandboxRandomWord(void);
int          sceKernelLoadStartModule(const char * name, std::size_t argc, const void * argv, std::uint32_t flags,
                                      const void * option, int * result);

int sceKeyboardInit(void);
int sceKeyboardOpen(int user, int type, int index, const void * param);
int sceKeyboardClose(int handle);
int sceKeyboardRead(int handle, ps5lm::RawKeyboardSample * samples, int capacity);
int sceKeyboardGetConnection(int handle, std::uint8_t * connected);

int sceMouseInit(void);
int sceMouseOpen(int user, int type, int index, const void * param);
int sceMouseClose(int handle);
int sceMouseRead(int handle, ps5lm::RawMouseData * data, int num);

int sceUserServiceInitialize(const void * params);
int sceUserServiceGetForegroundUser(std::int32_t * user);
int sceUserServiceGetInitialUser(std::int32_t * user);
int sceUserServiceGetLoginUserIdList(std::int32_t user_ids[4]);

int scePadInit(void);
int scePadOpen(int user, int type, int index, const void * parameters);
int scePadGetHandle(int user, int type, int index);
int scePadReadState(int handle, RawPadState * data);
int scePadClose(int handle);
}

template <typename Fn>
Fn read_got_fn(Fn sym) {
    Fn volatile slot = sym;
    return slot;
}

int load_system_prx(const char * sprx_name) {
    const char * word = sceKernelGetFsSandboxRandomWord();
    char         path[128];
    std::snprintf(path, sizeof(path), "/%s/common/lib/%s", word != nullptr ? word : "system", sprx_name);
    return sceKernelLoadStartModule(path, 0, nullptr, 0, nullptr, nullptr);
}

// Retail system SPRXs reject sceKernelDlsym by symbol name (0x80020003 / ESRCH), so we ensure
// the PRX is loaded and call through the NID-bound GOT entries.
template <typename F0, typename F1, typename F2, typename F3>
bool bind_prx_api(const char * label, const char * sprx_name, std::uint16_t sysmod_id, F0 sym0, F1 sym1, F2 sym2,
                  F3 sym3, int * handle_out, F0 * out0, F1 * out1, F2 * out2, F3 * out3, int * prx_rc_out,
                  int * sysmod_rc_out) {
    const int prx_rc = load_system_prx(sprx_name);
    int       sys_rc = 0;
    if (prx_rc < 0) {
        sys_rc = sceSysmoduleLoadModule(sysmod_id);
    }
    *prx_rc_out    = prx_rc;
    *sysmod_rc_out = sys_rc;

    F0 f0 = read_got_fn(sym0);
    F1 f1 = read_got_fn(sym1);
    F2 f2 = read_got_fn(sym2);
    F3 f3 = read_got_fn(sym3);
    if (f0 != nullptr && f1 != nullptr && f2 != nullptr && f3 != nullptr) {
        *handle_out = prx_rc >= 0 ? prx_rc : 0;
        *out0       = f0;
        *out1       = f1;
        *out2       = f2;
        *out3       = f3;
        return true;
    }

    ps5lm::log_line("usb-input: %s unavailable (prx=0x%08x sysmod=0x%08x got=%p)\n", label, (unsigned) prx_rc,
                    (unsigned) sys_rc, reinterpret_cast<const void *>(f0));
    return false;
}

template <typename Slot, typename Param, typename OpenFn>
void open_user_slots(const std::array<int, 4> & users, std::array<Slot, 4> & slots, OpenFn open_fn,
                     std::uint32_t err_invalid_arg, std::uint32_t err_already_opened, const char * kind,
                     const char * fn_name, std::uint32_t * last_err) {
    for (int uid : users) {
        if (uid == -1) {
            continue;
        }
        int  slot_idx = -1;
        bool already  = false;
        for (int i = 0; i < 4; ++i) {
            if (slots[static_cast<std::size_t>(i)].user_id == uid &&
                slots[static_cast<std::size_t>(i)].handle >= 0) {
                already = true;
                break;
            }
            if (slot_idx < 0 && slots[static_cast<std::size_t>(i)].handle < 0) {
                slot_idx = i;
            }
        }
        if (already || slot_idx < 0) {
            continue;
        }
        Param param{};
        int   h = open_fn(uid, 0, 0, &param);
        if (static_cast<std::uint32_t>(h) == err_invalid_arg) {
            h = open_fn(uid, 0, 0, nullptr);
        }
        if (h >= 0) {
            slots[static_cast<std::size_t>(slot_idx)] = { uid, h };
            *last_err                                 = 0;
            ps5lm::log_line("usb-input: %s opened user=0x%08x handle=%d\n", kind, (unsigned) uid, h);
        } else {
            const std::uint32_t err = static_cast<std::uint32_t>(h);
            if (err != err_already_opened && err != *last_err) {
                *last_err = err;
                ps5lm::log_line("usb-input: %s(user=0x%08x) -> 0x%08x\n", fn_name, (unsigned) uid, err);
            }
        }
    }
}

std::array<int, 4> collect_users() {
    std::array<int, 4> out{ -1, -1, -1, -1 };
    int                n   = 0;
    const auto         add = [&](int uid) {
        if (uid == -1 || uid == 0) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            if (out[static_cast<std::size_t>(i)] == uid) {
                return;
            }
        }
        if (n < 4) {
            out[static_cast<std::size_t>(n++)] = uid;
        }
    };

    std::int32_t list[4] = { -1, -1, -1, -1 };
    if (sceUserServiceGetLoginUserIdList(list) == 0) {
        for (int uid : list) {
            add(uid);
        }
    }
    std::int32_t fg = -1;
    if (sceUserServiceGetForegroundUser(&fg) == 0) {
        add(fg);
    }
    std::int32_t init_u = -1;
    if (sceUserServiceGetInitialUser(&init_u) == 0) {
        add(init_u);
    }
    return out;
}

}  // namespace

namespace ps5lm {

UsbInput::~UsbInput() {
    close();
}

void UsbInput::open() {
    if (opened_) {
        return;
    }
    opened_               = true;
    g_last_kbd_open_err   = 0;
    g_last_mouse_open_err = 0;

    const int us_rc = sceUserServiceInitialize(nullptr);
    if (us_rc != 0 && static_cast<std::uint32_t>(us_rc) != kErrAlreadyInitialized) {
        log_line("usb-input: sceUserServiceInitialize=0x%08x\n", (unsigned) us_rc);
    }

    int kbd_prx = 0;
    int kbd_sys = 0;
    if (bind_prx_api("libSceKeyboard", "libSceKeyboard.sprx", kSysmoduleKeyboard, &sceKeyboardInit, &sceKeyboardOpen,
                     &sceKeyboardClose, &sceKeyboardRead, &g_api.module_handle, &g_api.kbd_init, &g_api.kbd_open,
                     &g_api.kbd_close, &g_api.kbd_read, &kbd_prx, &kbd_sys)) {
        g_api.kbd_get_conn = read_got_fn(&sceKeyboardGetConnection);
        const int rc       = g_api.kbd_init();
        kbd_ready_         = (rc == 0 || static_cast<std::uint32_t>(rc) == kErrAlreadyInitialized);
        log_line("usb-input: keyboard ready via linked (prx=0x%08x sysmod=0x%08x handle=%d init=0x%08x)\n",
                 (unsigned) kbd_prx, (unsigned) kbd_sys, g_api.module_handle, (unsigned) rc);
    }

    int mouse_prx = 0;
    int mouse_sys = 0;
    if (bind_prx_api("libSceMouse", "libSceMouse.sprx", kSysmoduleMouse, &sceMouseInit, &sceMouseOpen, &sceMouseClose,
                     &sceMouseRead, &g_api.mouse_handle, &g_api.mouse_init, &g_api.mouse_open, &g_api.mouse_close,
                     &g_api.mouse_read, &mouse_prx, &mouse_sys)) {
        const int rc = g_api.mouse_init();
        mouse_ready_ = (rc == 0 || static_cast<std::uint32_t>(rc) == kErrAlreadyInitialized);
        log_line("usb-input: mouse ready via linked (prx=0x%08x sysmod=0x%08x handle=%d init=0x%08x)\n",
                 (unsigned) mouse_prx, (unsigned) mouse_sys, g_api.mouse_handle, (unsigned) rc);
    }

    const int pad_init = scePadInit();
    if (pad_init == 0 || static_cast<std::uint32_t>(pad_init) == kErrAlreadyInitialized) {
        int h = scePadOpen(kSystemUserId, kPadPortTypeRemoteControl, 0, nullptr);
        if (static_cast<std::uint32_t>(h) == kErrPadAlreadyOpened) {
            h = scePadGetHandle(kSystemUserId, kPadPortTypeRemoteControl, 0);
        }
        if (h >= 0) {
            remote_handle_ = h;
            log_line("usb-input: hdmi-cec remote opened handle=%d\n", remote_handle_);
        } else {
            log_line("usb-input: hdmi-cec remote not opened (0x%08x)\n", (unsigned) h);
        }
    }

    refresh_devices(0);
}

void UsbInput::refresh_devices(std::uint64_t now_us) {
    last_refresh_us_               = now_us;
    const std::array<int, 4> users = collect_users();

    if (kbd_ready_ && g_api.kbd_open) {
        for (int uid : users) {
            if (uid == -1) {
                continue;
            }
            for (int index = 0; index < 4; ++index) {
                int  slot_idx = -1;
                bool already  = false;
                for (int i = 0; i < KeyboardTracker::kMaxPorts; ++i) {
                    const KeyboardSlot & s = kbd_slots_[static_cast<std::size_t>(i)];
                    if (s.user_id == uid && s.index == index && s.handle >= 0) {
                        already = true;
                        break;
                    }
                    if (slot_idx < 0 && s.handle < 0) {
                        slot_idx = i;
                    }
                }
                if (already) {
                    continue;
                }
                if (slot_idx < 0) {
                    break;
                }

                int h = g_api.kbd_open(uid, 0, index, nullptr);
                if (index == 0 && static_cast<std::uint32_t>(h) == kErrKbdInvalidArg) {
                    RawKeyboardOpenParam param{};
                    h = g_api.kbd_open(uid, 0, index, &param);
                }
                if (h >= 0) {
                    KeyboardSlot & slot = kbd_slots_[static_cast<std::size_t>(slot_idx)];
                    slot                = { uid, index, h, false, false, 0 };
                    g_last_kbd_open_err = 0;
                    int          conn_rc     = -1;
                    std::uint8_t conn_buf[4] = {};
                    if (g_api.kbd_get_conn != nullptr) {
                        conn_rc = g_api.kbd_get_conn(h, conn_buf);
                        if (conn_rc == 0 && conn_buf[0] != 0) {
                            kbd_tracker_.set_port_connected(slot_idx, true);
                            slot.conn_state  = true;
                            slot.logged_conn = true;
                        }
                    }
                    log_line("usb-input: keyboard opened user=0x%08x idx=%d port=%d handle=%d conn_rc=0x%08x conn=%u\n",
                             (unsigned) uid, index, slot_idx, h, (unsigned) conn_rc, (unsigned) conn_buf[0]);
                } else {
                    const std::uint32_t err = static_cast<std::uint32_t>(h);
                    if (err == kErrKbdInvalidArg || err == kErrKbdInvalidPort || err == kErrDeviceInvalidIndex) {
                        break;
                    }
                    if (err != kErrKbdAlreadyOpened && err != g_last_kbd_open_err) {
                        g_last_kbd_open_err = err;
                        log_line("usb-input: sceKeyboardOpen(user=0x%08x idx=%d) -> 0x%08x\n", (unsigned) uid, index,
                                 err);
                    }
                }
            }
        }
    }

    if (mouse_ready_ && g_api.mouse_open) {
        open_user_slots<MouseSlot, RawMouseOpenParam>(
            users, mouse_slots_, g_api.mouse_open, kErrMouseInvalidArg, kErrMouseAlreadyOpened, "mouse",
            "sceMouseOpen", &g_last_mouse_open_err);
    }
}

void UsbInput::close() {
    for (int i = 0; i < KeyboardTracker::kMaxPorts; ++i) {
        KeyboardSlot & slot = kbd_slots_[static_cast<std::size_t>(i)];
        if (slot.handle >= 0 && g_api.kbd_close) {
            g_api.kbd_close(slot.handle);
        }
        slot = {};
        kbd_tracker_.clear_port(i);
    }
    for (int i = 0; i < MouseTracker::kMaxPorts; ++i) {
        MouseSlot & slot = mouse_slots_[static_cast<std::size_t>(i)];
        if (slot.handle >= 0 && g_api.mouse_close) {
            g_api.mouse_close(slot.handle);
        }
        slot = {};
        mouse_tracker_.clear_port(i);
    }
    if (remote_handle_ >= 0) {
        scePadClose(remote_handle_);
        remote_handle_ = -1;
    }
    g_last_kbd_open_err   = 0;
    g_last_mouse_open_err = 0;
    kbd_logged_on_        = false;
    mouse_logged_on_      = false;
    kbd_ready_            = false;
    mouse_ready_          = false;
    opened_               = false;
}

UsbFrame UsbInput::poll(std::uint64_t now_us, bool text_mode, bool pad_activity) {
    UsbFrame out;
    if (!opened_) {
        return out;
    }

    if (now_us >= last_refresh_us_ + 1'000'000) {
        refresh_devices(now_us);
    }

    RawKeyboardSample   kbd_buf[KeyboardTracker::kMaxPorts][16]{};
    KeyboardPortSamples batches[KeyboardTracker::kMaxPorts]{};
    int                 batch_count = 0;
    if (kbd_ready_ && g_api.kbd_read) {
        for (int i = 0; i < KeyboardTracker::kMaxPorts; ++i) {
            KeyboardSlot & slot = kbd_slots_[static_cast<std::size_t>(i)];
            if (slot.handle < 0) {
                continue;
            }
            const int rc = g_api.kbd_read(slot.handle, kbd_buf[i], 16);
            if (rc > 0) {
                const int n        = std::min(rc, 16);
                slot.last_read_err = 0;
                for (int j = 0; j < n; ++j) {
                    const RawKeyboardSample & s   = kbd_buf[i][j];
                    const bool                con = s.connected != 0;
                    if (!slot.logged_conn || con != slot.conn_state) {
                        slot.logged_conn = true;
                        slot.conn_state  = con;
                        log_line("usb-input: keyboard port=%d (idx=%d) %s count=%u\n", i, slot.index,
                                 con ? "connected" : "disconnected", s.count);
                    }
                    if (con && s.intercepted == 0 && !kbd_logged_on_) {
                        std::uint16_t first_key = 0;
                        for (int k = 0; k < kMaxKeyboardKeySlots; ++k) {
                            if (s.keycodes[k] >= kHidMinValid && s.keycodes[k] <= kHidMaxValid) {
                                first_key = s.keycodes[k];
                                break;
                            }
                        }
                        if (first_key != 0) {
                            kbd_logged_on_ = true;
                            log_line("usb-input: keyboard active port=%d idx=%d mods=0x%x locks=0x%x key=0x%02x\n", i,
                                     slot.index, s.modifiers, s.locks, (unsigned) first_key);
                        }
                    }
                }
                batches[batch_count++] = { i, std::span<const RawKeyboardSample>(kbd_buf[i],
                                                                                 static_cast<std::size_t>(n)) };
            } else if (rc < 0) {
                const std::uint32_t err = static_cast<std::uint32_t>(rc);
                if (err == kErrKbdInvalidHandle) {
                    if (g_api.kbd_close) {
                        g_api.kbd_close(slot.handle);
                    }
                    slot = {};
                    kbd_tracker_.clear_port(i);
                } else if (err != slot.last_read_err) {
                    slot.last_read_err = err;
                    log_line("usb-input: sceKeyboardRead(port=%d handle=%d) -> 0x%08x\n", i, slot.handle, err);
                }
            }
        }
    }

    hui::InputFrame kbd_actions;
    bool            kbd_activity = false;
    kbd_tracker_.update(std::span<const KeyboardPortSamples>(batches, static_cast<std::size_t>(batch_count)), now_us,
                        text_mode, &kbd_actions, &out.text, &kbd_activity);

    RawMouseData     mouse_buf[MouseTracker::kMaxPorts][64]{};
    MousePortSamples mouse_batches[MouseTracker::kMaxPorts]{};
    int              mouse_batch_count = 0;
    if (mouse_ready_ && g_api.mouse_read) {
        for (int i = 0; i < MouseTracker::kMaxPorts; ++i) {
            MouseSlot & slot = mouse_slots_[static_cast<std::size_t>(i)];
            if (slot.handle < 0) {
                continue;
            }
            const int rc = g_api.mouse_read(slot.handle, mouse_buf[i], 64);
            if (rc > 0) {
                const int n = std::min(rc, 64);
                for (int j = 0; j < n; ++j) {
                    if (mouse_buf[i][j].connected != 0 && !mouse_logged_on_) {
                        mouse_logged_on_ = true;
                        log_line("usb-input: mouse active user=0x%08x buttons=0x%x dx=%d dy=%d wheel=%d\n",
                                 (unsigned) slot.user_id, mouse_buf[i][j].buttons, mouse_buf[i][j].x_axis,
                                 mouse_buf[i][j].y_axis, mouse_buf[i][j].wheel);
                    }
                }
                mouse_batches[mouse_batch_count++] = {
                    i, std::span<const RawMouseData>(mouse_buf[i], static_cast<std::size_t>(n))
                };
            } else if (rc < 0 && static_cast<std::uint32_t>(rc) == kErrMouseInvalidHandle) {
                if (g_api.mouse_close) {
                    g_api.mouse_close(slot.handle);
                }
                slot = {};
                mouse_tracker_.clear_port(i);
            }
        }
    }

    hui::InputFrame mouse_actions;
    const bool      kbd_act     = kbd_activity || kbd_actions.nav != hui::Direction::none || kbd_actions.pressed != 0 ||
                                  !out.text.empty();
    const bool      hide_cursor = pad_activity || kbd_act;
    mouse_tracker_.update(
        std::span<const MousePortSamples>(mouse_batches, static_cast<std::size_t>(mouse_batch_count)), now_us,
        hide_cursor, &mouse_actions, &out.pointer);

    RawRemoteSample remote_sample{};
    if (remote_handle_ >= 0) {
        RawPadState raw{};
        if (scePadReadState(remote_handle_, &raw) == 0 && raw.connected != 0 &&
            (raw.buttons & hui::pad_bits::kIntercepted) == 0) {
            remote_sample.buttons   = raw.buttons;
            remote_sample.connected = true;
            remote_sample.cec_code  = raw.reserved1[3];
        }
    }
    const hui::InputFrame remote_actions = remote_tracker_.update(remote_sample, now_us);

    out.actions            = merge_input(merge_input(kbd_actions, mouse_actions), remote_actions);
    out.keyboard_connected = kbd_tracker_.connected();
    out.mouse_connected    = mouse_tracker_.connected();
    out.keyboard_activity  = kbd_act;
    out.mouse_activity     = out.pointer.moved || out.pointer.left_pressed || out.pointer.left_released ||
                             out.pointer.right_pressed || out.pointer.middle_pressed || out.pointer.wheel != 0;
    out.keyboard_active    = out.keyboard_connected;
    out.mouse_active       = out.pointer.visible || out.mouse_activity;
    return out;
}

}  // namespace ps5lm

#else

namespace ps5lm {

UsbInput::~UsbInput() = default;
void     UsbInput::open() {}
void     UsbInput::refresh_devices(std::uint64_t) {}
void     UsbInput::close() {}
UsbFrame UsbInput::poll(std::uint64_t, bool, bool) { return {}; }

}  // namespace ps5lm

#endif
