import math
import pandas as pd
import numpy as np
import scipy as sc

diary = pd.read_csv('./src/data/diary.csv', sep=';')
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


# Расчет Reach 1+ по объединенной аудитории нескольких станций
def RN_mm(id_list, aud, day, n, n_hours):
    df = aud
    r1 = rt[rt['ID'].isin(id_list)]
    sample_size = r1['w'].sum()
    df1 = df[df['day'] == day]
    col_list = spot_list(n_hours)
    df1['sum'] = df1.loc[:, col_list].sum(axis=1) * n / 4
    df2 = df1.groupby('Member_nr').sum().reset_index()
    df2['w'] = df2['weights'] / (df2['day'] / day)
    df2['E'] = df2['sum'].apply(cut)
    Rn_m = np.dot(df2['w'], df2['E']) * 7 / sample_size
    aud_size = (sample_size / 15138) * 10973.3
    return Rn_m, Rn_m * aud_size, aud_size, sample_size


# Расчет дневного Reach 1+ по объединенной аудитории нескольких станций
def RN_m(id_list, aud, day, n, n_hours):
    df = aud
    r1 = rt[rt['ID'].isin(id_list)]
    sample_size = r1['w'].sum()
    df1 = df[df['day'] == day]
    col_list = spot_list(n_hours)
    df1.loc[:, 'sum'] = df1.loc[:, col_list].sum(axis=1) * n / 4
    df2 = df1.groupby(['Member_nr', 'weights']).sum().reset_index()
    df2['E'] = df2['sum'].apply(cut)
    Rn_m = np.dot(df2['weights'], df2['E']) * 7 / sample_size
    aud_size = (sample_size / 15138) * 10973.3
    return Rn_m, Rn_m * aud_size, aud_size, sample_size


# Итоговый недельный Reach по одной станции
def WR(id_list, ch, days_list, n_hours, n):
    aud = diary[diary['Member_nr'].isin(id_list)]
    # Расчет WR
    lst = []
    lst.append(ch)
    WR, sample_size, aud_size = weekly_reach(id_list, lst)

    # print ('WR', WR, sample_size, aud_size)

    # Формирование  аудитории
    fin_aud = aud[aud['Chl_id'] == ch]

    # Расчет кэмпэйн ричей
    reach_c = []
    for d in (days_list):
        r_c, *r = RN(id_list, ch, d, n_hours, n)
        reach_c.append(r_c)
    total_c = sum(reach_c)

    # Подсчет RCR
    tc = 1
    for d_r in reach_c:
        tc = tc * (1 - d_r / WR)
    RCR = 1 - tc
    # print ('RCR', RCR)

    # Подсчет RSR и максимального поправочного коэф-нта к расчетам по RCR
    reach_s_max = []
    c_l = []
    for i in range(0, 24):
        c_l.append(i)
    for day in range(1, 8):
        r_s, *rs = RN_m(id_list, fin_aud, day, 50, c_l)
        reach_s_max.append(r_s)
    total_s_max = sum(reach_s_max)
    ts1 = 1
    for dr in reach_s_max:
        ts1 = ts1 * (1 - dr / WR)
    RSR = 1 - ts1
    ratio_cf = 1 / RSR
    # print ('ratio_cf', ratio_cf, 'total_c/total_s_max', total_c/total_s_max)

    # Скалирование коэф-нта к RCR
    fin_ratio_cf = ((ratio_cf - 1) * (total_c / total_s_max)) + 1
    # if (0.2<(total_c/total_s_max)<=0.4):
    #  R_m=fin_ratio_cf*RCR*WR*1.0015
    # elif (0.4<(total_c/total_s_max)<=0.6):
    #  R_m=fin_ratio_cf*RCR*WR*1.003
    # elif (0.6<(total_c/total_s_max)<=0.8):
    #  R_m=fin_ratio_cf*RCR*WR*1.0015
    # else:
    R_m = fin_ratio_cf * RCR * WR
    # print ('R_m', R_m, 'fin_ratio_cf', fin_ratio_cf, 'RSR', RSR)
    # return fin_ratio_cf*1.0014*RCR*WR*10973.3

    # TRP
    trp_list = []
    for ds in days_list:
        trp_list.append(GRP_m(fin_aud, sample_size, ds, n, n_hours))
    total_trp = sum(trp_list)

    # GRP
    grp_list = []
    for ds in days_list:
        grp_list.append(GRP(diary, ds, ch, n, n_hours))
    total_grp = sum(grp_list)

    a, k = parNBD(R_m, total_trp / 100)
    # print (a,k)
    # Rn=dict()
    # print ('Spots ', n*len(n_hours)*len(days_list))
    R_fin = dict()
    R_fin['Spots'] = n * len(n_hours) * len(days_list)
    R_fin['TRP'] = total_trp
    R_fin['GRP'] = total_grp
    R_fin['Reach 1+ '] = R_m
    R_fin['Reach, 000, 1+ '] = R_m * aud_size
    for i in range(1, 6):
        # Rn['Reach %s '% i ]=NBD (i, a, k)
        r = i + 1
        R_fin['Reach %s+ ' % r] = R_fin['Reach %s+ ' % i] - NBD(i, a, k)
        R_fin['Reach, 000, %s+ ' % r] = R_fin['Reach %s+ ' % r] * aud_size
    # return fin_ratio_cf*RCR*WR
    return R_fin


# Итоговый недельный Reach по нескольким станциям
def WR_cum(id_list, ch_list, days_list, n_hours, n):
    # Расчет WR
    WR, sample_size, aud_size = weekly_reach(id_list, ch_list)

    # Формирование аудитории станций, потом фильтр по ЦА
    first_aud = diary[diary['Chl_id'] == ch_list[0]]
    for ch in (ch_list):
        if ch != ch_list[0]:
            sec_aud = diary[diary['Chl_id'] == ch]
            cum_aud = pd.concat([first_aud, sec_aud], ignore_index=True)
            first_aud = cum_aud
    fin_aud = first_aud[first_aud['Member_nr'].isin(id_list)]

    # Расчет кэмпэйн ричей
    reach_c = []
    for d in (days_list):
        r_c, *r = RN_m(id_list, fin_aud, d, n, n_hours)
        reach_c.append(r_c)
    total_c = sum(reach_c)
    # print (reach_c)

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
        r_s, *rs = RN_m(id_list, fin_aud, day, 50, c_l)
        reach_s_max.append(r_s)
    total_s_max = sum(reach_s_max)
    ts1 = 1
    for dr in reach_s_max:
        ts1 = ts1 * (1 - dr / WR)
    RSR = 1 - ts1
    ratio_cf = 1 / RSR

    # Скалирование коэф-нта к RCR
    fin_ratio_cf = ((ratio_cf - 1) * (total_c / total_s_max)) + 1
    R_m = fin_ratio_cf * RCR * WR

    # TRP
    trp_list = []
    for ds in days_list:
        trp_list.append(GRP_m(fin_aud, sample_size, ds, n, n_hours))
    total_trp = sum(trp_list)

    # GRP
    grp_list = []
    for ds in days_list:
        grp_list.append(GRP_m(first_aud, 15138, ds, n, n_hours))
    total_grp = sum(grp_list)
    print('GRP ', total_grp, 'TRP ', total_trp, 'Reach', R_m)

    a, k = parNBD(R_m, total_trp / 100)

    # Rn=dict()
    print('Spots ', n * len(n_hours) * len(days_list) * len(ch_list))
    R_fin = dict()
    R_fin['Spots'] = n * len(n_hours) * len(days_list) * len(ch_list)
    R_fin['TRP'] = round(total_trp, 2)
    R_fin['GRP'] = round(total_grp, 2)
    R_fin['Reach% 1+ '] = round(R_m, 4)
    # R_fin['Reach, 000, 1+ ']=R_m*aud_size
    for i in range(1, 6):
        # Rn['Reach %s '% i ]=NBD (i, a, k)
        r = i + 1
        R_fin[f'Reach% {r}+ '] = round(R_fin[f'Reach% {i}+ '] - NBD(i, a, k), 4)
    for r in range(1, 6):
        R_fin[f'Reach {r}+ '] = round(R_fin[f'Reach% {r}+ '] * aud_size, 4)
    return R_fin


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


# Расчет накопленной суммы рейтингов (параметры: day - день, ch -номер станции, n - кол-во роликов в час, *n_hours - час(ы), с которых стартует размещение)
def GRP(aud, ch, day, n, *n_hours):
    df = aud
    col_list = []
    for h in n_hours:
        col_list.extend(spot_list(h))
    # df1=df[(mask1|mask2|mask3|mask4|mask5)&(df['Chl_id']==ch)]
    # df1=df[df['Chl_id']==ch]
    df1 = df[(df['day'] == day) & (df['Chl_id'] == ch)]
    rat_list = []
    for slot in col_list:
        ser = df1[str(slot)] * (n / 4) * df1['weights']
        rat_list.append(ser.sum())
    return sum(rat_list) / (15138 / 7) * 100


# Расчет накопленной суммы рейтингов по всем дням (параметры: ch -номер станции, n - кол-во роликов в час, *n_hours - час(ы), с которых стартует размещение)
def all_GRP(aud, ch, n, *n_hours):
    df = aud
    col_list = []
    for h in n_hours:
        col_list.extend(spot_list(h))
    # mask1=df['day']==day1
    # mask2=df['day']==day2
    # mask3=df['day']==day3
    # mask4=df['day']==day4
    # mask5=df['day']==day5
    # df1=df[(mask1|mask2|mask3|mask4|mask5)&(df['Chl_id']==ch)]
    df1 = df[df['Chl_id'] == ch]
    # df1=df[(df['day']==day)&(df['Chl_id']==ch)]
    rat_list = []
    for slot in col_list:
        ser = df1[slot] * (n / 4) * df['weights']
        rat_list.append(ser.sum())
    return (sum(rat_list) / 15138) * 100


# Расчет Reach n на основе методики DAR - через суммирование личных рейтингов слушания респондентов (параметры: day - день, ch -номер станции, n - кол-во роликов в час, *n_hours - час(ы), с которых стартует размещение)
def RN_DAR(id_list, day, ch, n, *n_hours):
    df = diary[diary['Member_nr'].isin(id_list)]
    df1 = df[(df['day'] == day) & (df['Chl_id'] == ch)]
    col_list = spot_list(*n_hours)
    # print (len(n_hours[0]))
    df1['sum'] = df1[col_list].sum(axis=1) * n / 4
    Rn = dict()
    ln = n * len(n_hours[0])
    if ln > 5: ln = 5
    for r in range(1, ln + 1):
        df1['Prb %s' % r] = df1['sum'] - (r - 1)
        df1['Reach %s' % r] = df1['Prb %s' % r].apply(cut)
        Rn['Reach %s + ' % r] = np.dot(df1['weights'], df1['Reach %s' % r]) * 7 * (10973.3 / 15138)
        # Rn['Reach %s'% r ]=np.dot (df1['weights'], df1['Reach %s'% r ])
    return Rn


# Расчет Reach n на основе методики DAR - округление через cut sharp (параметры: day - день, ch -номер станции, n - кол-во роликов в час, *n_hours - час(ы), с которых стартует размещение)
def RN_cut_sharp(aud, day, ch, n, *n_hours):
    df = aud
    # mask1=df['day']==day1
    # mask2=df['day']==day2
    # mask3=df['day']==day3
    # mask4=df['day']==day4
    # mask5=df['day']==day5
    # df1=df[(mask1|mask2|mask3|mask4|mask5)&(df['Chl_id']==ch)]
    # df1=df[df['Chl_id']==ch]
    df1 = df[(df['day'] == day) & (df['Chl_id'] == ch)]
    col_list = spot_list(*n_hours)
    # print (col_list)
    df1['sum'] = df1[col_list].sum(axis=1) * n / 4
    Rn = dict()
    for r in range(1, n + 1):
        df1['Prb %s' % r] = df1['sum'] - (r - 1)
        df1['Reach %s' % r] = df1['Prb %s' % r].apply(cut_sharp)
        # Rn['Reach %s'% r ]=np.dot (df1['weights'], df1['Reach %s'% r ])*7*(10973.3/15138)
        Rn['Reach %s' % r] = np.dot(df1['weights'], df1['Reach %s' % r])
    return Rn


# Расчет вероятности событий по распределению Пуассона (параметры: n - кол-во слотов, l - сумма рейтингов плана)
def pois(n, l):
    for x in range(1, n + 1):
        c = pow(np.exp(1), -x)
        p = pow(l, x) * c / math.factorial(x)
        print('p', x, p * 10973.3)


# Расчет вероятности событий по распределению NBD (параметры: i - частота контакта, a, k - параметры распределения полученные через функцию rNBD )
def NBD(i, a, k):
    g = math.gamma(k + i) / (math.gamma(k) * math.gamma(i + 1))
    p = g * pow(1 / (1 + a), k) * pow(a / (1 + a), i)
    return p


# Функция для вычисления парамеров a и k для NBD-распределения перебором (параметры: Reach - Reach 1+ в виде дроби (%/100), GRP - накопленный GRP/TRP в МП в виде дроби (%/100))
# def pNBD (Reach, Rat) :
#  p0=round (np.log(1-Reach), 4)
#  a_list=[]
#  p_list=[]
#  for a in range (1,5000000,1):
#      a1=a/100000
#      k1=Rat/a1
#      z=round (-k1*(np.log(1+a1)),4)
#      if p0==z:
#        a_list.append (a1)
#        p_list.append (k1)
#  alpha=sum(a_list)/len(a_list)
#  k=sum(p_list)/len(p_list)
#  return  alpha,  k

# Итоговая модель расчета Reach n+ (параметры: day - день, ch -номер станции, n - кол-во роликов в час, *n_hours - час(ы), с которых стартует размещение)
# def ReachNBD (aud, day, ch, n, *n_hours):
#  rc=RN (aud, day, ch, n, *n_hours).get('Reach 1+', 'Reach is not calculated by RN () func')/10973.3
#  rt=GRP (aud, day, ch, n, *n_hours)/100
#  a, k = parNBD (Reach_day=rc, Rat_day=rt)
#  rb=dict ()
#  for r in range (0,n+1):
#    rb['p(%s)'% r ]=NBD (r, a=a, k=k)*10973.3
#  return rb




# Функция для вычисления парамеров a и k для NBD-распределения (параметры: Reach - Reach 1+ в виде дроби (%/100), GRP - накопленный GRP/TRP в МП в виде дроби (%/100))
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


# Расчет WR, Sample size, Aud size для ЦА
def weekly_reach(id_list, ch_list):
    # Расчет size ЦА
    r1 = rt[rt['ID'].isin(id_list)]
    ss = r1['w'].sum()
    aus = (ss / 15138) * 10973.3

    # Расчет WR по станций
    raw_list = []
    for ch in ch_list:
        df_raw = rt[rt['week_list'] != 0]
        raw_list.extend(df_raw['ID'].values)
    df1 = rt[rt['ID'].isin(raw_list)]
    df2 = df1[df1['ID'].isin(id_list)]
    tw = df2['w'].sum()
    return tw / ss, ss, aus
