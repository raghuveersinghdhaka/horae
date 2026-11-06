/* One slot of the HORAE device loop in single precision, with every arithmetic operation counted.
 * Prints the operation counts of an active and of an inactive device and the size of the device state.
 * The counters are compiled out with -DNO_COUNT, which leaves the loop as it would run on a device.
 * With -DRULE_R the loop adds the hazard threshold and the controller guard of HORAE-R. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef NO_COUNT
#define OP(k)
#else
static unsigned long c_add, c_mul, c_div, c_cmp, c_exp, c_log, c_rng;
#define OP(k) (c_##k++)
#endif

typedef struct {            /* what a device must keep from slot to slot */
    uint32_t age;           /* A_i, slots since the last confirmed delivery */
    float held;             /* last value confirmed delivered */
    float sent;             /* reading of the last attempt */
    float beta;             /* entry factor */
    uint32_t rng;           /* xorshift32 state */
    uint16_t last_res;      /* resource used in the last attempt */
    uint8_t  attempted;     /* attempted in the last slot */
} dev_state;

typedef struct {            /* configuration, may live in flash */
    float w, theta, rho, kappa, boost, v0, delta, eta;
    uint16_t S;
#ifdef RULE_R
    float thr, beta_max;    /* hazard threshold lambda N/S and upper limit of the entry factor */
#endif
} dev_config;

static uint32_t xs32(uint32_t *s) {
    uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x;
    OP(rng);
    return x;
}

static float unif(uint32_t *s) { uint32_t x = xs32(s); OP(mul); return (float)x * 2.3283064e-10f; }

/* alert goal: activity test (eq. active), significance score (eq. value), hazard, probability, attempt */
static int contend(dev_state *d, const dev_config *c, float x, uint16_t *res) {
    float e = x - c->theta; OP(add);
    float ae = fabsf(e); OP(cmp);
    float band = c->kappa * c->delta; OP(mul);
    int near = ae < band; OP(cmp);
    float he = d->held - c->theta; OP(add);
    int wrong = (e > 0.0f) != (he > 0.0f); OP(cmp); OP(cmp);
    if (!(near || wrong)) return 0;
    float v = c->kappa * expf(-ae / c->rho); OP(div); OP(exp); OP(mul);
    if (wrong) { v += c->boost; OP(add); }
    if (v < c->v0) v = c->v0; OP(cmp);
    float h = c->w * v * (float)d->age; OP(mul); OP(mul);
#ifdef RULE_R
    OP(cmp);
    if (!wrong && h < c->thr) return 0;   /* a report that would change the answer is exempt */
#endif
    float p = 1.0f - expf(-d->beta * h); OP(mul); OP(exp); OP(add);
    if (unif(&d->rng) >= p) { OP(cmp); return 0; }
    OP(cmp);
    *res = (uint16_t)(xs32(&d->rng) % c->S); OP(div);
    d->sent = x;
    return 1;
}

/* beacon: idle count for the entry factor, outcome symbol of the device's own resource */
static void on_beacon(dev_state *d, const dev_config *c, int n_idle, const uint8_t *map) {
    float idle = (float)n_idle; float fl = c->S >= 2 ? 0.5f : 0.25f;
    if (idle < fl) idle = fl; OP(cmp);
    float f = idle / (float)c->S; OP(div);
    if (f > 0.999f) f = 0.999f; OP(cmp);
    float Gh = -(float)c->S * logf(f); OP(log); OP(mul);
    float err = logf((float)c->S) - logf(Gh); OP(log); OP(log); OP(add);
#ifdef RULE_R
    OP(cmp);
    if (n_idle == 0) { err *= 3.0f; OP(mul); }
    OP(cmp);
    if (n_idle < c->S) { d->beta *= expf(c->eta * err); OP(mul); OP(exp); OP(mul); }
    if (d->beta > c->beta_max) d->beta = c->beta_max; OP(cmp);
#else
    d->beta *= expf(c->eta * err); OP(mul); OP(exp); OP(mul);
    if (d->beta > 50.0f) d->beta = 50.0f; OP(cmp);
#endif
    if (d->beta < 1e-12f) d->beta = 1e-12f; OP(cmp);
    int own_success = 0;
    OP(cmp);
    if (d->attempted) {     /* the symbol of the own resource: five ternary symbols to a byte */
        unsigned v = map[d->last_res / 5]; OP(div);
        unsigned k = d->last_res % 5; OP(div);
        while (k--) { v /= 3; OP(div); OP(cmp); }
        own_success = (v % 3) == 1; OP(div); OP(cmp);
    }
    if (own_success) { d->age = 1; d->held = d->sent; } else { d->age += 1; OP(add); }
    OP(cmp);
}

#ifndef NO_COUNT
static void show(const char *label) {
    unsigned long basic = c_add + c_mul + c_div + c_cmp;
    printf("%s add %lu mul %lu div %lu cmp %lu exp %lu log %lu rng %lu basic %lu transcendental %lu\n",
           label, c_add, c_mul, c_div, c_cmp, c_exp, c_log, c_rng, basic, c_exp + c_log);
    c_add = c_mul = c_div = c_cmp = c_exp = c_log = c_rng = 0;
}
#endif

int main(void) {
    dev_config c = {1.0f, 100.0f, 8.0f, 3.0f, 4.0f, 0.05f, 4.0f, 0.25f, 21
#ifdef RULE_R
                    , 2.0f*480.0f/21.0f, 0.25f*21.0f/480.0f
#endif
    };
    dev_state d;
    uint16_t r = 0;
    memset(&d, 0, sizeof d); d.age = 40; d.held = 90.0f; d.beta = 0.01f; d.rng = 2463534242u;
    uint8_t map[20];
    memset(map, 0, sizeof map);
    /* a quiet source: not near the boundary, held view on the correct side */
    int a = contend(&d, &c, 60.0f, &r); d.attempted = (uint8_t)a; on_beacon(&d, &c, 0, map);
#ifndef NO_COUNT
    show("inactive");
#endif
    /* the costliest slot: held view on the wrong side, the draw won, the packet lost, the longest symbol lookup */
    d.beta = 50.0f; d.age = 40;
#ifdef RULE_R
    /* with no idle resource, so that the guard's factor of three is counted */
    a = contend(&d, &c, 101.0f, &r); d.attempted = (uint8_t)a; d.last_res = 4; on_beacon(&d, &c, 0, map);
#else
    a = contend(&d, &c, 101.0f, &r); d.attempted = (uint8_t)a; d.last_res = 4; on_beacon(&d, &c, 8, map);
#endif
#ifndef NO_COUNT
    show("active");
#endif
    printf("state_bytes %u config_bytes %u attempted %d res %u beta %g\n",
           (unsigned)sizeof(dev_state), (unsigned)sizeof(dev_config), a, (unsigned)r, (double)d.beta);
    return 0;
}
