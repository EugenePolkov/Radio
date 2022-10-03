import math
import pandas as pd
import numpy as np
import scipy as sc

diary = pd.read_csv('src/data/diary_full.csv', sep=';')
rt = pd.read_csv('./src/data/data.csv', sep=';')


# Main functions

# Расчет рейтинга 15-минутки
def reach_slot(day, ch, slot):
    aud = diary[(diary['day'] == day) & (diary['Chl_id'] == ch)]
    return np.dot(aud['weights'].values, aud[slot].values) * 7 / 15138


# расчет GRP (wrapper)
def grp(ch, days_list, n_hours, n):
    aud = diary[diary['Chl_id'] == ch]
    # GRP по дням
    grp_list = []
    for ds in days_list:
        grp_list.append(GRP_m(aud, 15138, ds, n, n_hours))
    total_grp = sum(grp_list)
    return total_grp


# Базовый расчет Reach 1+ по одному дню и станции на основе методики DAR - через суммирование личных рейтингов слушания респондентов
def reach_day(ch, day, n_hours, n):
    aud = diary[(diary['Chl_id'] == ch) & (diary['day'] == day)]
    col_list = [str(x) for x in spot_list(n_hours)]
    aud['sum'] = aud.loc[:, col_list].sum(axis=1) * n / 4
    aud['E'] = aud['sum'].apply(cut)
    return np.dot(aud['weights'], aud['E']) * 7 / 15138


# Расчет охвата по нескольким дням
def mix_reach(ch, days_list, n_hours, n):
    # Расчет WR
    WR = np.dot(np.where(rt['week_list'] == 0, 0, 1), rt['w']) / 15138

    # Расчет кэмпэйн ричей
    reach_c = []
    for day in (days_list):
        r_c = reach_day(ch, day, n_hours, n)
        reach_c.append(r_c)
    total_c = sum(reach_c)

    # Подсчет RCR
    tc = 1
    for d_r in reach_c:
        tc = tc * (1 - d_r / WR)
    RCR = 1 - tc

    # Подсчет RSR и максимального поправочного коэф-нта к расчетам по RCR
    reach_s_max = []
    c_l = []
    for i in range(0, 24):
        c_l.append(i)
    for day in range(1, 8):
        r_s = reach_day(ch, day, c_l, 12)
        reach_s_max.append(r_s)
    total_s_max = sum(reach_s_max)
    ts1 = 1
    for dr in reach_s_max:
        ts1 = ts1 * (1 - dr / WR)
    RSR = 1 - ts1
    ratio_cf = 1 / RSR

    # Скалирование коэф-нта к RCR
    fin_ratio_cf = ((ratio_cf - 1) * (total_c / total_s_max)) + 1
    return fin_ratio_cf * RCR * WR


# Частотное распределение охвата по NBD
def reach_NBD(reach, grp):
    a, k = parNBD(reach, grp)
    R_fin = dict()
    td= dict()
    td['Reach% 1+ '] = reach
    for i in range(1, 5):
        r = i + 1
        td[f'Reach% {r}+ '] = td[f'Reach% {i}+ '] - NBD(i, a, k)
    for i in range(1, 6):
        R_fin[f'Reach {i}+ '] = round(td[f'Reach% {i}+ '] * 10973.3, 2)
    for i in range(1, 6):
        R_fin[f'Reach% {i}+ '] = round(td[f'Reach% {i}+ '] * 100, 2)
    return R_fin


# Оформление итогового МП
def mp(ch, days_list, n_hours, n):
    mp_d = dict()
    reach = mix_reach(ch, days_list, n_hours, n)
    gr = grp(ch, days_list, n_hours, n)
    a, k = parNBD(reach, gr / 100)
    mp_d['GI'] = round (gr * 10973.3 / 100 , 2 )
    mp_d['Frequency'] = round ((gr /(reach*100)) , 2)
    mp_d['Index T/U Reach'] = 100*reach/reach # в нашем случае в числителе охват по всей аудитории, так как ЦА = все
    mp_d['GRP'] = round (gr , 2)
    mp_d['TRP'] = round (gr , 2)  # в нашем случае TRP=GRP
    mp_d['RP Index'] = 100 * gr / gr  # также в нашем случае в числителе GRP вместо TRP
    mp_d['Spots'] = n * len(n_hours) * len(days_list)
    td= dict()
    td['Reach% 1+ '] = reach
    for i in range(1, 5):
        r = i + 1
        td[f'Reach% {r}+ '] = td[f'Reach% {i}+ '] - NBD(i, a, k)
    for i in range(1, 6):
        mp_d[f'Reach {i}+ '] = round(td[f'Reach% {i}+ '] * 10973.3, 2)
    for i in range(1, 6):
        mp_d[f'Reach% {i}+ '] = round(td[f'Reach% {i}+ '] * 100, 2)
    return pd.DataFrame.from_dict (mp_d, orient='index', columns=['Расчет'])

def GRP_m(aud, sample_size, day, n, *n_hours):
    df = aud
    col_list = []
    for h in n_hours:
        col_list.extend(spot_list(h))
    df1 = df[df['day'] == day]
    rat_list = []
    for slot in col_list:
        ser = df1[slot] * (n / 4) * df['weights']
        rat_list.append(ser.sum())
    return sum(rat_list) * 7 / sample_size * 100


# Расчет вероятности событий по распределению NBD (параметры: i - частота контакта, a, k - параметры распределения полученные через функцию parNBD )
def NBD(i, a, k):
    g = math.gamma(k + i) / (math.gamma(k) * math.gamma(i + 1))
    p = g * pow(1 / (1 + a), k) * pow(a / (1 + a), i)
    return p


# Функция для вычисления парамеров a и k для NBD-распределения (параметры: Reach - Reach 1+ в виде дроби (%/100), GRP - накопленный TRP в МП в виде дроби (%/100))
def parNBD(Reach_day, Rat_day):
    a = sc.optimize.root_scalar(f=lambda x: ((-Rat_day / np.log(1 - Reach_day)) * np.log(1 + x) - x),
                                bracket=[1e-12, 1000], xtol=1e-6,
                                fprime=lambda x: (-Rat_day / np.log(1 - Reach_day)) / (1 + x) - 1, x0=0.001,
                                method='ridder').root
    k = Rat_day / a
    return (a, k)


# вспомогательные функции
# получение номеров слотов из часов
def spot_list(n_hours):
    l_s = []
    for i in range(5, 24):
        l_s.append(i)
    for k in range(0, 5):
        l_s.append(k)
    spot_n = dict()
    for ix, l in enumerate(l_s):
        tl = [ix * 4 + 1, ix * 4 + 2, ix * 4 + 3, ix * 4 + 4]
        spot_n[l] = tl
    col_list = []
    for n in n_hours:
        col_list.extend(spot_n.get(n))
    col_list_s = [str(x) for x in col_list]
    return col_list_s


# обрезка значений до 1, если значение больше 1
def cut(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x


# критерий для выбора распределения
def crit(reach, grp):
    return grp / np.log(1 - reach)
