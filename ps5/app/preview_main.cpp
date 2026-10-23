// PC preview of the PS5LM dashboard: the same drawing code as the console,
// rendered off-screen through Mesa's surfaceless EGL to PNG files, with
// sample data shaped like a real run (Qwen 3.8 27B at 64k on the GPU).
//
//   scripts/preview-app-ui.sh            writes build/app-preview/*.png
//
// SPDX-License-Identifier: GPL-3.0-or-later
#include "dashboard.hpp"

#include "core/save_file.hpp"
#include "gfx/gl_program.hpp"
#include "gfx/renderer.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool open_context() {
    auto get_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    EGLDisplay display = get_display ? get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr)
                                     : eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0, minor = 0;
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) || !eglBindAPI(EGL_OPENGL_API)) {
        return false;
    }
    const EGLint attrs[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5,
                             EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE };
    EGLContext context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
    return context != EGL_NO_CONTEXT && eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
}

bool load_font(hui::gfx::Renderer & r, const std::string & path, hui::gfx::Font * font, hui::ui::FontRef * ref) {
    std::string data;
    if (!hui::save::read_file(path, &data) || !font->load(data)) {
        std::fprintf(stderr, "cannot load font %s\n", path.c_str());
        return false;
    }
    ref->font    = font;
    ref->texture = r.batch().create_font_texture(*font);
    return true;
}

// A run as the console measured it: Qwen3.8-27B UD-Q2_K_XL, 64k, q4_0 cache.
float g_loading = -1;  // >= 0: draw the loading state at this progress
bool  g_decision = false;  // d1 Omni 600M instead of Qwen 3.8
bool  g_empty = false;     // nothing loaded
std::string g_phase;       // the app still starting

ps5lm::Live sample(float seconds, bool generating) {
    ps5lm::Live L;
    L.state       = generating ? ps5lm::ServerState::generating : ps5lm::ServerState::ready;
    L.address     = "192.168.1.19:8081";
    L.model_label = "Qwen3.8-27B-UD-Q2_K_XL";
    L.model_arch  = "qwen35";
    L.preset      = "Qwen 3.8";
    L.model_gib   = 9.15;
    L.ctx         = 65536;
    L.kv_type     = "q4_0";
    L.kv_gib      = 1.13;
    L.model_file  = "/data/PS5LM/models/Qwen3.8-27B-UD-Q2_K_XL.gguf";
    L.plan        = "64k context, q4_0 cache: 11.3 of 11.4 GiB";
    L.args = { "-m", L.model_file, "-ngl", "999", "-fit", "off", "-lm", "none", "-c", "65536", "-fa", "on",
               "-ctk", "q4_0", "-ctv", "q4_0", "-b", "256", "-ub", "64", "-np", "1", "-ctxcp", "1", "-cram", "0",
               "--host", "0.0.0.0", "--port", "8081", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
               "--min-p", "0", "--metrics" };
    L.soc_temp    = 61;
    L.cpu_temp    = 54;
    L.soc_power_w = 86;
    L.energy_wh   = 0.42;
    L.fan         = 0.23f;
    L.temps = { { "cpu", 54 }, { "soc0", 61 }, { "soc1", 60 }, { "soc2", 59 }, { "soc3", 62 }, { "soc4", 61 },
                { "soc5", 60 }, { "soc6", 58 } };
    L.cpu_ghz     = 3.5f;
    L.drives = { { "Internal storage", "/data/PS5LM/models", 8, 825 }, { "M.2 drive", "/mnt/ext0/PS5LM/models", 1520, 1863 }, { "USB drive", "/mnt/usb0/PS5LM/models", 455, 477 } };
    L.pool_gib    = 11.44;
    L.free_gib    = 0.31;
    L.heap_gib    = 0.76;
    L.cpus        = 13;
    L.uptime_s    = 754 + seconds;
    L.tokens      = 4182;
    L.prompt_tokens = 9120;
    L.requests    = 14;
    L.served      = 14;
    L.ctx_used    = 2318;
    L.prompt_tps  = 45.4f;
    for (int i = 0; i < 120; ++i) {
        const bool on = (i / 14) % 3 != 0 || i > 104;
        const float tps = on ? 20.4f + 0.6f * std::sin(i * 0.7f) : 0.0f;
        L.gen_history.push_back(tps);
        L.gpu_history.push_back(on ? 0.92f + 0.05f * std::sin(i * 0.3f) : 0.03f);
        L.cpu_history.push_back(on ? 0.11f + 0.03f * std::sin(i * 0.5f) : 0.02f);
        L.temp_history.push_back(on ? 60.0f + 2.0f * std::sin(i * 0.2f) : 52.0f);
        L.power_history.push_back(on ? 85.0f + 1.5f * std::sin(i * 0.4f) : 23.0f);
        // An hour: idle, then a long session with the 27B from minute 20.
        const bool busy = i > 40;
        L.temp_hour.push_back(busy ? 58.0f + 3.0f * std::sin(i * 0.15f) : 50.0f + 0.5f * std::sin(i * 0.4f));
        L.power_hour.push_back(busy ? 70.0f + 12.0f * std::sin(i * 0.23f) : 24.0f);
        L.gpu_hour.push_back(busy ? 0.7f + 0.2f * std::sin(i * 0.31f) : 0.02f);
        L.cpu_hour.push_back(busy ? 0.1f + 0.03f * std::sin(i * 0.5f) : 0.02f);
    }
    {
        // Requests from the network: two answered, the newest still streaming.
        ps5lm::Exchange a;
        a.id = 12; a.client = "192.168.1.31"; a.path = "/v1/messages"; a.status = 200; a.seconds = 12.8;
        a.asked = "Rename the loop variable in parse_args to something clearer and explain why";
        a.answered = "Renamed i to arg_index: it says what the loop walks, and the next reader does not have to look up what i counts.";
        L.activity.push_back(a);
        a.id = 13; a.client = "192.168.1.24"; a.path = "/v1/chat/completions"; a.status = 200; a.seconds = 3.4;
        a.asked = "Write a haiku about a console that runs a language model";
        a.answered = "Fans hum through the night, a console dreams in tokens, words bloom on the TV.";
        L.activity.push_back(a);
        a.id = 14; a.client = "192.168.1.24"; a.path = "/v1/chat/completions"; a.status = generating ? 0 : 200; a.seconds = 9.1;
        a.asked = "How does the PS5's GPU run llama.cpp?";
        a.answered = "Through Vulkan. Mihawk-99's RADV port gives the console a full Vulkan 1.4 driver, so llama.cpp's own Vulkan "
                     "backend runs unchanged: every matrix product becomes a compute shader that ACO compiles on the console, and "
                     "the weights sit in the 12 GB the GPU and CPU share";
        L.activity.push_back(a);
    }
    L.gen_tps  = generating ? 20.7f : 0.0f;
    L.gpu_busy = generating ? 0.94f : 0.02f;
    L.cpu_use  = generating ? 0.12f : 0.02f;
    const char * files[] = { "Qwen3.8-27B-UD-IQ1_S", "Qwen3.8-27B-UD-IQ2_S", "Qwen3.8-27B-UD-IQ2_XXS",
                             "Qwen3.8-27B-UD-Q2_K_XL", "gpt-oss-20b-MXFP4" };
    const char * plans[] = { "64k context, q8_0 cache: 8.9 of 11.0 GiB", "64k context, q8_0 cache: 11.0 of 11.0 GiB",
                             "64k context, q8_0 cache: 9.9 of 11.0 GiB", "64k context, q4_0 cache: 11.3 of 11.4 GiB",
                             "needs 11.6 GiB with a 4k context; 11.0 GiB is free" };
    const double sizes[] = { 5.77, 7.80, 6.77, 9.15, 11.3 };
    for (int i = 0; i < 5; ++i) {
        ps5lm::ModelRow m;
        m.file = std::string("/data/PS5LM/models/") + files[i] + ".gguf";
        m.drive = "Internal storage";
        m.label = files[i];
        m.plan = plans[i];
        m.size_gib = sizes[i];
        m.fits = i < 4;
        m.current = i == 3;
        m.preset = i < 4 ? "Qwen 3.8" : "gpt-oss";
        L.models.push_back(m);
    }
    ps5lm::ModelRow usb;
    usb.file = "/mnt/usb0/PS5LM/models/Ornith-1.5-9B-Q4_K_M.gguf";
    usb.label = "Ornith-1.5-9B-Q4_K_M";
    usb.plan = "64k context, q8_0 cache: 8.6 of 11.0 GiB";
    usb.preset = "Ornith 1.5";
    usb.size_gib = 5.38;
    usb.fits = true;
    usb.drive = "USB drive";
    L.models.push_back(usb);
    usb.file = "/mnt/usb0/PS5LM/models/gemma-4-12b-it-Q3_K_M.gguf";
    usb.label = "gemma-4-12b-it-Q3_K_M";
    usb.plan = "64k context, q8_0 cache: 9.1 of 11.0 GiB";
    usb.preset = "Gemma 4";
    usb.size_gib = 5.30;
    L.models.push_back(usb);
    L.models[0].decision = "";
    {
        ps5lm::ModelRow d;
        d.file = "/data/PS5LM/models/d1-omni-600M-BF16.gguf";
        d.label = "d1-omni-600M-BF16";
        d.plan = "decision model (lfm2-d1-omni), 4k context, f16 cache, images: 2.9 of 11.0 GiB";
        d.decision = "lfm2-d1-omni";
        d.size_gib = 0.71;
        d.fits = true;
        d.drive = "Internal storage";
        L.models.push_back(d);
    }
    if (g_decision) {
        for (auto & m : L.models) {
            m.current = m.decision == "lfm2-d1-omni";
        }
        L.state = generating ? ps5lm::ServerState::generating : ps5lm::ServerState::ready;
        L.model_label = "d1-omni-600M-BF16";
        L.model_arch = "lfm2";
        L.decision = "lfm2-d1-omni";
        L.preset = "";
        L.model_gib = 0.71;
        L.ctx = 4096;
        L.ctx_used = 793;
        L.prompt_tps = 2140.0f;
        L.gen_tps = 0;
        L.tokens = 0;
        L.prompt_tokens = 88412;
        L.requests = 131;
        L.free_gib = 8.6;
        L.activity.clear();
        ps5lm::Exchange a;
        a.id = 130; a.client = "192.168.1.40"; a.path = "/v1/systemone"; a.status = 200; a.seconds = 0.3; a.questions = 3;
        a.asked = "3 questions: {\"from\": \"promo@cheap-meds.biz\", \"subject\": \"URGENT: you won a free iPhone!!!\"}";
        a.answered = "spam yes 1.00, intent sell, urgency 1.02";
        L.activity.push_back(a);
        a.id = 131; a.asked = "7 questions, 1 image"; a.answered = "text no 0.01, color black, ad no 0.01, people no 0.00, mood 1.00";
        a.seconds = 2.9; a.questions = 7; L.activity.push_back(a);
    }
    if (g_empty) {
        L.state = ps5lm::ServerState::no_model;
        L.model_label.clear();
        L.decision.clear();
        L.requests = L.tokens = L.prompt_tokens = 0;
        for (auto & m : L.models) {
            m.current = false;
        }
    }
    if (g_loading >= 0) {
        L.state = ps5lm::ServerState::loading;
        L.load_progress = g_loading;
        L.phase = g_phase;
        L.free_gib = 11.03 - g_loading * (L.model_gib + L.kv_gib);
        L.gen_tps = 0;
        L.gpu_busy = 0;
    }
    return L;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <kit assets dir> <output dir> <app assets dir>\n", argv[0]);
        return 2;
    }
    const std::string assets = argv[1], out = argv[2], app = argv[3];
    const int W = 1920, H = 1080;
    if (!open_context()) {
        std::fprintf(stderr, "no surfaceless EGL OpenGL 4.5 context\n");
        return 1;
    }
    hui::gfx::set_glsl_prefix("#version 450 core\n");
    hui::gfx::Renderer renderer;
    hui::gfx::Font regular, semibold, display, mono, pixel, hand;
    hui::ui::Fonts fonts;
    if (!renderer.init() || !load_font(renderer, app + "/fonts/manrope-500.huifont", &regular, &fonts.regular) ||
        !load_font(renderer, app + "/fonts/manrope-600.huifont", &semibold, &fonts.semibold) ||
        !load_font(renderer, app + "/fonts/manrope-200.huifont", &display, &fonts.display) ||
        !load_font(renderer, assets + "/fonts/dejavu-sans-mono.huifont", &mono, &fonts.mono)) {
        return 1;
    }
    fonts.pixel = fonts.mono;
    fonts.hand  = fonts.regular;

    GLuint fb = 0, color = 0;
    glGenFramebuffers(1, &fb);
    glGenRenderbuffers(1, &color);
    glBindRenderbuffer(GL_RENDERBUFFER, color);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);

    hui::save::ensure_directory(out);
    ps5lm::Dashboard dash(fonts);
    ps5lm::Settings settings;
    settings.default_model = "/data/PS5LM/models/Qwen3.8-27B-UD-Q2_K_XL.gguf";
    dash.set_settings(settings);
    std::vector<std::string> app_log, llama_log;
    for (int i = 0; i < 60; ++i) {
        app_log.push_back("[ps5-present-perf] calls=2500 failures=0 warmup_frames=30 idle_ms=0.000898 flip_ms=0.000889");
        app_log.push_back("22:56:01 ps5lm-app: model Qwen3.8-27B-UD-IQ2_XXS.gguf (qwen35): 64k context, q8_0 cache: 9.9 of 11.0 GiB");
        llama_log.push_back("7.15.902.312 I slot print_timing: id  0 | task 646 | n_gen =    162, tg =  20.44 t/s");
    }
    app_log.push_back("22:58:14 ps5lm-app: accept failed, errno 53");
    app_log.push_back("W srv          stop: cancel task, id_task = 22182");
    dash.set_logs(app_log, llama_log);
    ps5lm::MarketView mv;
    const char * ids[] = { "unsloth/Qwen3.8-27B-GGUF", "unsloth/gemma-4-E4B-it-GGUF", "bartowski/Llama-3.2-3B-Instruct-GGUF",
                           "ggml-org/gpt-oss-20b-GGUF", "unsloth/Qwen3.5-9B-GGUF", "bartowski/Mistral-7B-Instruct-v0.3-GGUF" };
    for (int i = 0; i < 6; ++i) {
        ps5lm::MarketRepo r;
        r.id = ids[i];
        r.downloads = 812000 / (i + 1);
        mv.repos.push_back(r);
    }
    mv.open = 0;
    const char * fnames[] = { "Qwen3.8-27B-UD-IQ2_XXS.gguf", "Qwen3.8-27B-UD-Q2_K_XL.gguf", "Qwen3.8-27B-Q3_K_M.gguf", "Qwen3.8-27B-Q4_K_M.gguf" };
    const double fsz[] = { 6.8, 9.2, 12.6, 15.4 };
    const int ffits[] = { 1, 1, 0, 0 };
    const char * fplans[] = { "64k context, q8_0 cache: 9.9 of 11.0 GiB", "32k context, q4_0 cache: 10.8 of 11.0 GiB",
                              "too big: the weights alone need more than is free", "too big: the weights alone need more than is free" };
    for (int i = 0; i < 4; ++i) {
        ps5lm::MarketFile f;
        f.name = fnames[i];
        f.bytes = (uint64_t) (fsz[i] * 1073741824.0);
        f.fits = ffits[i];
        f.plan = fplans[i];
        mv.repos[0].files.push_back(f);
    }
    {
        ps5lm::Download d;
        d.repo = "unsloth/Qwen3.8-27B-GGUF";
        d.file = "Qwen3.8-27B-UD-IQ2_XXS.gguf";
        d.dir = "/data/PS5LM/models";
        d.bytes = (uint64_t) (6.8 * 1073741824.0);
        d.done = (uint64_t) (2.5 * 1073741824.0);
        d.state = ps5lm::DownloadState::running;
        d.mbps = 14.2;
        mv.downloads.push_back(d);
        d.file = "gemma-4-12b-it-Q4_K_M.gguf";
        d.repo = "unsloth/gemma-4-12b-it-GGUF";
        d.dir = "/mnt/usb0/PS5LM/models";
        d.bytes = (uint64_t) (7.1 * 1073741824.0);
        d.done = 0;
        d.mbps = 0;
        d.state = ps5lm::DownloadState::queued;
        mv.downloads.push_back(d);
        d.file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";
        d.repo = "bartowski/Llama-3.2-3B-Instruct-GGUF";
        d.bytes = 2019377696;
        d.done = 812000000;
        d.state = ps5lm::DownloadState::retrying;
        d.note = "Connection lost (sceHttp2ReadData 0x804101a3); trying again in 12 s, attempt 2";
        mv.downloads.push_back(d);
        d.file = "SmolLM2-135M-Instruct-F16.gguf";
        d.repo = "unsloth/SmolLM2-135M-Instruct-GGUF";
        d.dir = "/data/PS5LM/models";
        d.bytes = d.done = 270885952;
        d.state = ps5lm::DownloadState::done;
        d.note = "Saved to /data/PS5LM/models/SmolLM2-135M-Instruct-F16.gguf";
        mv.downloads.push_back(d);
    }
    dash.set_market(mv);
    std::vector<unsigned char> pixels((size_t) W * H * 4);
    stbi_flip_vertically_on_write(1);
    hui::ui::Feedback feedback;
    float t = 0;
    bool ok = true;

    const auto step = [&](int frames, hui::InputFrame input, bool generating) {
        for (int i = 0; i < frames; ++i) {
            dash.set_live(sample(t, generating));
            feedback.clear();
            dash.update(i == 0 ? input : hui::InputFrame{}, 1.0f / 60.0f, feedback);
            t += 1.0f / 60.0f;
        }
    };
    const auto shot = [&](const char * name) {
        ps5lm::DashboardFrame frame;
        frame.glass_texture = renderer.glass_texture();
        dash.draw(frame);
        renderer.begin();
        renderer.backdrop(frame.backdrop);
        renderer.draw(frame.scene);
        if (frame.glass) {
            renderer.glass();
        }
        renderer.draw(frame.overlay);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.present(fb, W, H);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        for (size_t i = 3; i < pixels.size(); i += 4) {
            pixels[i] = 255;
        }
        const std::string path = out + "/" + name + ".png";
        ok = stbi_write_png(path.c_str(), W, H, 4, pixels.data(), W * 4) != 0 && ok;
        std::fprintf(stderr, "wrote %s: %zu shapes, %zu draw calls, GL error 0x%x\n", path.c_str(),
                     renderer.last_instances(), renderer.last_draw_calls(), glGetError());
    };
    auto press = [](hui::Action a) {
        hui::InputFrame in;
        in.connected = true;
        in.pressed = hui::action_bit(a);
        return in;
    };
    auto nav = [](hui::Direction d) {
        hui::InputFrame in;
        in.connected = true;
        in.nav = d;
        return in;
    };

    using D = hui::Direction;
    using A = hui::Action;
    g_loading = 0;
    g_phase = "Opening the GPU and reading the models";
    step(50, {}, false);
    shot("00-home-starting");
    g_phase.clear();
    g_loading = 0.45f;
    step(90, {}, false);
    shot("00-home-loading");
    g_loading = -1;
    step(120, {}, true);
    shot("01-home");
    step(60, {}, false);
    shot("02-home-idle");
    g_decision = true;
    step(60, {}, true);
    shot("03-home-decision");
    g_decision = false;
    g_empty = true;
    step(60, {}, false);
    shot("04-home-empty");
    g_empty = false;
    step(40, press(A::confirm), false);
    shot("05-models-local");
    step(20, press(A::back), false);
    step(20, nav(D::down), false);
    step(20, press(A::confirm), false);
    step(20, nav(D::down), false);
    step(30, nav(D::down), false);
    shot("06-models-suggested");
    step(20, press(A::back), false);
    step(20, nav(D::down), false);
    step(20, press(A::confirm), false);
    step(30, nav(D::down), false);
    shot("07-models-search");
    step(30, press(A::confirm), false);
    dash.set_market(mv);
    step(20, nav(D::down), false);
    shot("08-models-files");
    step(30, press(A::confirm), false);
    shot("09-models-space");
    step(20, press(A::back), false);
    step(20, press(A::back), false);
    step(20, press(A::back), false);
    step(20, nav(D::down), false);
    step(20, press(A::confirm), false);
    step(30, nav(D::down), false);
    shot("10-models-downloads");
    step(40, press(A::page_next), false);
    shot("11-api");
    g_decision = true;
    step(40, {}, false);
    shot("12-api-decision");
    g_decision = false;
    step(40, press(A::page_next), false);
    shot("13-console");
    step(30, press(A::jump_next), false);
    shot("14-console-hour");
    step(40, press(A::page_next), false);
    shot("15-settings");
    step(20, press(A::confirm), false);
    step(30, nav(D::down), false);
    shot("16-settings-row");
    step(30, press(A::confirm), false);
    shot("17-settings-picker");
    step(20, press(A::back), false);
    step(20, press(A::back), false);
    step(20, nav(D::down), false);
    step(20, press(A::confirm), false);
    step(30, press(A::confirm), false);
    shot("18-settings-context");
    step(20, press(A::back), false);
    step(20, press(A::back), false);
    for (int i = 0; i < 2; ++i) {
        step(15, nav(D::down), false);
    }
    // Sound and storage, on a console where the HEN kept /data closed.
    dash.set_storage(false, 16, false);
    step(30, press(A::confirm), false);
    step(20, nav(D::down), false);
    step(20, nav(D::down), false);
    shot("18-settings-app-storage");
    step(20, press(A::back), false);
    dash.set_storage(true, 16, false);
    step(15, nav(D::down), false);
    step(30, press(A::confirm), false);
    shot("19-settings-report");
    step(30, press(A::confirm), false);
    shot("19-settings-report-dialog");
    step(20, press(A::back), false);
    dash.set_report_status(false, "Saved to /data/PS5LM/debug-report.txt and /mnt/usb0/PS5LM-debug-report.txt, 214 KB");
    step(20, {}, false);
    shot("19-settings-report-saved");
    step(20, press(A::back), false);
    step(15, nav(D::down), false);
    step(30, press(A::confirm), false);
    shot("19-settings-diagnostics");
    step(20, press(A::back), false);
    for (int i = 0; i < 4; ++i) {
        step(15, press(A::page_prev), false);
    }
    step(30, press(A::north), true);
    step(60, {}, true);
    shot("20-home-message");
    step(30, press(A::back), true);
    // The first start without full storage.
    dash.set_storage(false, 16, true);
    step(30, {}, false);
    shot("20-storage-closed");
    step(20, press(A::north), false);
    shot("20-storage-howto");
    step(20, press(A::back), false);
    step(20, press(A::confirm), false);
    dash.set_busy("Loading Qwen3.8 27B", "Restarting with the whole memory for it, about 90 seconds");
    step(30, {}, false);
    shot("21-busy");
    return ok ? 0 : 1;
}
