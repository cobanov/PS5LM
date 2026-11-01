/* A payload that starts an installed title, for scripts/ps5lm-app.sh launch:
 * elfldr runs it, it asks the system to launch TITLE_ID for the foreground
 * user and exits. elfldr passes no arguments, so the title ID is built in
 * (scripts/build-app.sh makes build/app-ps5/launch-<TITLE_ID>.elf).
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef TITLE_ID
#error "build with -DTITLE_ID=\"PPSA99575\""
#endif

typedef struct {
    uint32_t sz;
    uint32_t user_id;
    uint32_t app_opt;
    uint64_t crash_report;
    uint32_t check_flag;
} LncAppParam;

int sceUserServiceInitialize(void *);
int sceUserServiceGetForegroundUser(int *);
int sceLncUtilLaunchApp(const char *, const char **, LncAppParam *);

int main(void) {
    int user = -1;
    sceUserServiceInitialize(0);
    if (sceUserServiceGetForegroundUser(&user) != 0) {
        puts("launch-app: no foreground user");
        return 1;
    }
    LncAppParam param;
    memset(&param, 0, sizeof param);
    param.sz = sizeof param;
    param.user_id = (uint32_t) user;
    int rc = sceLncUtilLaunchApp(TITLE_ID, NULL, &param);
    printf("launch-app: %s -> 0x%08x\n", TITLE_ID, (unsigned) rc);
    return rc < 0;
}
