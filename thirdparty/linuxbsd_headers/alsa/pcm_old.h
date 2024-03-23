/*
 * Old ALSA 0.9.x API
 */

#ifdef ALSA_PCM_OLD_HW_PARAMS_API

asm(".symver snd_pcm_hw_params_get_access,snd_pcm_hw_params_get_access@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_access_first,snd_pcm_hw_params_set_access_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_access_last,snd_pcm_hw_params_set_access_last@ALSA_0.9");

int snd_pcm_hw_params_get_access(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_test_access(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t val);
int snd_pcm_hw_params_set_access(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t val);
snd_pcm_access_t snd_pcm_hw_params_set_access_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
snd_pcm_access_t snd_pcm_hw_params_set_access_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_set_access_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_mask_t *mask);
void snd_pcm_hw_params_get_access_mask(snd_pcm_hw_params_t *params, snd_pcm_access_mask_t *mask);

asm(".symver snd_pcm_hw_params_get_format,snd_pcm_hw_params_get_format@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_format_first,snd_pcm_hw_params_set_format_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_format_last,snd_pcm_hw_params_set_format_last@ALSA_0.9");

int snd_pcm_hw_params_get_format(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_test_format(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t val);
int snd_pcm_hw_params_set_format(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t val);
snd_pcm_format_t snd_pcm_hw_params_set_format_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
snd_pcm_format_t snd_pcm_hw_params_set_format_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_set_format_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_mask_t *mask);
void snd_pcm_hw_params_get_format_mask(snd_pcm_hw_params_t *params, snd_pcm_format_mask_t *mask);

asm(".symver snd_pcm_hw_params_get_subformat,snd_pcm_hw_params_get_subformat@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_subformat_first,snd_pcm_hw_params_set_subformat_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_subformat_last,snd_pcm_hw_params_set_subformat_last@ALSA_0.9");

int snd_pcm_hw_params_test_subformat(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t val);
int snd_pcm_hw_params_get_subformat(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_set_subformat(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t val);
snd_pcm_subformat_t snd_pcm_hw_params_set_subformat_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
snd_pcm_subformat_t snd_pcm_hw_params_set_subformat_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_set_subformat_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_mask_t *mask);
void snd_pcm_hw_params_get_subformat_mask(snd_pcm_hw_params_t *params, snd_pcm_subformat_mask_t *mask);

asm(".symver snd_pcm_hw_params_get_channels,snd_pcm_hw_params_get_channels@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_channels_min,snd_pcm_hw_params_get_channels_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_channels_max,snd_pcm_hw_params_get_channels_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_channels_near,snd_pcm_hw_params_set_channels_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_channels_first,snd_pcm_hw_params_set_channels_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_channels_last,snd_pcm_hw_params_set_channels_last@ALSA_0.9");

int snd_pcm_hw_params_get_channels(const snd_pcm_hw_params_t *params);
unsigned int snd_pcm_hw_params_get_channels_min(const snd_pcm_hw_params_t *params);
unsigned int snd_pcm_hw_params_get_channels_max(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_test_channels(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_set_channels(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_set_channels_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, unsigned int *max);
unsigned int snd_pcm_hw_params_set_channels_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
unsigned int snd_pcm_hw_params_set_channels_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
unsigned int snd_pcm_hw_params_set_channels_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

asm(".symver snd_pcm_hw_params_get_rate,snd_pcm_hw_params_get_rate@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_rate_min,snd_pcm_hw_params_get_rate_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_rate_max,snd_pcm_hw_params_get_rate_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_rate_near,snd_pcm_hw_params_set_rate_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_rate_first,snd_pcm_hw_params_set_rate_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_rate_last,snd_pcm_hw_params_set_rate_last@ALSA_0.9");

int snd_pcm_hw_params_get_rate(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_rate_min(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_rate_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_rate(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_rate(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_rate_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
unsigned int snd_pcm_hw_params_set_rate_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int *dir);
unsigned int snd_pcm_hw_params_set_rate_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_set_rate_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_set_rate_resample(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_get_rate_resample(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);

asm(".symver snd_pcm_hw_params_get_period_time,snd_pcm_hw_params_get_period_time@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_period_time_min,snd_pcm_hw_params_get_period_time_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_period_time_max,snd_pcm_hw_params_get_period_time_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_time_near,snd_pcm_hw_params_set_period_time_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_time_first,snd_pcm_hw_params_set_period_time_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_time_last,snd_pcm_hw_params_set_period_time_last@ALSA_0.9");

int snd_pcm_hw_params_get_period_time(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_period_time_min(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_period_time_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_period_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_period_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_period_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
unsigned int snd_pcm_hw_params_set_period_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int *dir);
unsigned int snd_pcm_hw_params_set_period_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_set_period_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);

asm(".symver snd_pcm_hw_params_get_period_size,snd_pcm_hw_params_get_period_size@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_period_size_min,snd_pcm_hw_params_get_period_size_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_period_size_max,snd_pcm_hw_params_get_period_size_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_size_near,snd_pcm_hw_params_set_period_size_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_size_first,snd_pcm_hw_params_set_period_size_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_period_size_last,snd_pcm_hw_params_set_period_size_last@ALSA_0.9");

snd_pcm_sframes_t snd_pcm_hw_params_get_period_size(const snd_pcm_hw_params_t *params, int *dir);
snd_pcm_uframes_t snd_pcm_hw_params_get_period_size_min(const snd_pcm_hw_params_t *params, int *dir);
snd_pcm_uframes_t snd_pcm_hw_params_get_period_size_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_period_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val, int dir);
int snd_pcm_hw_params_set_period_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val, int dir);
int snd_pcm_hw_params_set_period_size_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *min, int *mindir, snd_pcm_uframes_t *max, int *maxdir);
snd_pcm_uframes_t snd_pcm_hw_params_set_period_size_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val, int *dir);
snd_pcm_uframes_t snd_pcm_hw_params_set_period_size_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
snd_pcm_uframes_t snd_pcm_hw_params_set_period_size_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_set_period_size_integer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

asm(".symver snd_pcm_hw_params_get_periods,snd_pcm_hw_params_get_periods@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_periods_min,snd_pcm_hw_params_get_periods_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_periods_max,snd_pcm_hw_params_get_periods_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_periods_near,snd_pcm_hw_params_set_periods_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_periods_first,snd_pcm_hw_params_set_periods_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_periods_last,snd_pcm_hw_params_set_periods_last@ALSA_0.9");

int snd_pcm_hw_params_get_periods(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_periods_min(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_periods_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_periods(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_periods(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_periods_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
unsigned int snd_pcm_hw_params_set_periods_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int *dir);
unsigned int snd_pcm_hw_params_set_periods_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_set_periods_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_set_periods_integer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

asm(".symver snd_pcm_hw_params_get_buffer_time,snd_pcm_hw_params_get_buffer_time@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_buffer_time_min,snd_pcm_hw_params_get_buffer_time_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_buffer_time_max,snd_pcm_hw_params_get_buffer_time_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_time_near,snd_pcm_hw_params_set_buffer_time_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_time_first,snd_pcm_hw_params_set_buffer_time_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_time_last,snd_pcm_hw_params_set_buffer_time_last@ALSA_0.9");

int snd_pcm_hw_params_get_buffer_time(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_buffer_time_min(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_buffer_time_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_buffer_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_buffer_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_buffer_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
unsigned int snd_pcm_hw_params_set_buffer_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int *dir);
unsigned int snd_pcm_hw_params_set_buffer_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_set_buffer_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);

asm(".symver snd_pcm_hw_params_get_buffer_size,snd_pcm_hw_params_get_buffer_size@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_buffer_size_min,snd_pcm_hw_params_get_buffer_size_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_buffer_size_max,snd_pcm_hw_params_get_buffer_size_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_size_near,snd_pcm_hw_params_set_buffer_size_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_size_first,snd_pcm_hw_params_set_buffer_size_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_buffer_size_last,snd_pcm_hw_params_set_buffer_size_last@ALSA_0.9");

snd_pcm_sframes_t snd_pcm_hw_params_get_buffer_size(const snd_pcm_hw_params_t *params);
snd_pcm_uframes_t snd_pcm_hw_params_get_buffer_size_min(const snd_pcm_hw_params_t *params);
snd_pcm_uframes_t snd_pcm_hw_params_get_buffer_size_max(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_test_buffer_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_hw_params_set_buffer_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_hw_params_set_buffer_size_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *min, snd_pcm_uframes_t *max);
snd_pcm_uframes_t snd_pcm_hw_params_set_buffer_size_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_hw_params_set_buffer_size_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
snd_pcm_uframes_t snd_pcm_hw_params_set_buffer_size_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

asm(".symver snd_pcm_hw_params_get_tick_time,snd_pcm_hw_params_get_tick_time@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_tick_time_min,snd_pcm_hw_params_get_tick_time_min@ALSA_0.9");
asm(".symver snd_pcm_hw_params_get_tick_time_max,snd_pcm_hw_params_get_tick_time_max@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_tick_time_near,snd_pcm_hw_params_set_tick_time_near@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_tick_time_first,snd_pcm_hw_params_set_tick_time_first@ALSA_0.9");
asm(".symver snd_pcm_hw_params_set_tick_time_last,snd_pcm_hw_params_set_tick_time_last@ALSA_0.9");

int snd_pcm_hw_params_get_tick_time(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_tick_time_min(const snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_get_tick_time_max(const snd_pcm_hw_params_t *params, int *dir);
int snd_pcm_hw_params_test_tick_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_tick_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_tick_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_tick_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_tick_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
unsigned int snd_pcm_hw_params_set_tick_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int *dir);
unsigned int snd_pcm_hw_params_set_tick_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);
unsigned int snd_pcm_hw_params_set_tick_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, int *dir);

#endif /* ALSA_PCM_OLD_HW_PARAMS_API */


#ifdef ALSA_PCM_OLD_SW_PARAMS_API

asm(".symver snd_pcm_sw_params_get_tstamp_mode,snd_pcm_sw_params_get_tstamp_mode@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_sleep_min,snd_pcm_sw_params_get_sleep_min@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_avail_min,snd_pcm_sw_params_get_avail_min@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_xfer_align,snd_pcm_sw_params_get_xfer_align@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_start_threshold,snd_pcm_sw_params_get_start_threshold@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_stop_threshold,snd_pcm_sw_params_get_stop_threshold@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_silence_threshold,snd_pcm_sw_params_get_silence_threshold@ALSA_0.9");
asm(".symver snd_pcm_sw_params_get_silence_size,snd_pcm_sw_params_get_silence_size@ALSA_0.9");

int snd_pcm_sw_params_set_tstamp_mode(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_tstamp_t val);
snd_pcm_tstamp_t snd_pcm_sw_params_get_tstamp_mode(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_sleep_min(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, unsigned int val);
unsigned int snd_pcm_sw_params_get_sleep_min(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_avail_min(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_avail_min(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_xfer_align(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_xfer_align(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_start_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_start_threshold(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_stop_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_stop_threshold(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_silence_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_silence_threshold(const snd_pcm_sw_params_t *params);
int snd_pcm_sw_params_set_silence_size(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
snd_pcm_uframes_t snd_pcm_sw_params_get_silence_size(const snd_pcm_sw_params_t *params);

#endif /* ALSA_PCM_OLD_SW_PARAMS_API */
