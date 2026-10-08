"""
opus_scorer_v22.py — перший прохід відбору моментів (рубрика docs/selection-standard.md v2.2, вересень 2026).

Що нового проти v2 (калібрування на 154 шортсах редакції «Що з економікою?», 57 випусків, 03.09.2026):
  • Cold-open скасовано (selection-standard v2.1): найцитованіший рядок дає бал як «панчлайн є», а не як старт.
    У редакції він стоїть у середньому на 30 % довжини кліпу (≈15 с від старту), на старті — лише у 18 %.
  • Старт = перше змістовне слово, а не початок речення: редакція ріже лід-ін (філери, «я думаю, що»,
    «ну дивіться», «у нас») — медіана 3 слова, 76 % стартів усередині речення. lead_trim до 8 слів.
  • Тест предмета (перші 6 слів: іменник/число/питання) — +4 / −6 у Hook (тест контексту docs/selection-standard.md).
  • Довжина: пік 50 с (референс: медіана 51, p25–p75 = 41–63), понад --softmax 80 с — штраф.
  • Зміна спікера (маркер >> в авто-субтитрах YouTube): ≥ 2 зміни всередині кліпу — штраф у Flow
    (у редакції один спікер; ведучий — лише сетап-питанням).
  • --teaser N: перші N секунд випуску (тизер-монтаж) не беремо в кандидати.

Вхід: words.json — [{"w":слово,"t":старт_с[,"e":кінець_с]}, ...] (Whisper або YouTube-авто-субтитри з пунктуацією).
Запуск:
  python opus_scorer_v22.py words.json --top 40 --teaser 60 --out cands_v22.json --md cands_v22.md
Це ПЕРШИЙ прохід (тріаж). Фінальний відбір і межі — за людиною по docs/selection-standard.md.

Калібрування на референсі (перекриття ≥ 40 % їхнього кліпу; «центр» = центр мого кліпа всередині їхнього):
  v2   топ-30: 76 % шортсів (85 % за переглядами) | довжина мій/їхній 1.46 | центр@30 60 %
  v2.2 топ-30: 62 % (64 %)  топ-40: 78 % (86 %)  | довжина 1.08 | центр@30 58 %  → дефолт --top 40
  Тобто моменти знаходимо однаково, вікна стали правильної довжини; лексика НЕ передбачає вибір редакції
  (логістична регресія по всіх під-балах: +2 п.п. recall) — вибір моменту робить людина/Claude за змістом.
"""
import sys, json, re, argparse, math

ap = argparse.ArgumentParser()
ap.add_argument('words')
ap.add_argument('--min', type=float, default=40)
ap.add_argument('--max', type=float, default=120)
ap.add_argument('--peak', type=float, default=50)
ap.add_argument('--softmax', type=float, default=80)
ap.add_argument('--teaser', type=float, default=0)   # перші N с випуску = тизер-монтаж, кандидатів там не будуємо   # понад це — штраф у Flow (референс: p75 = 63 с, макс 101)
ap.add_argument('--top', type=int, default=40)
ap.add_argument('--out', default='cands_v22.json')
ap.add_argument('--md', default='')
a = ap.parse_args()

W = json.load(open(a.words, encoding='utf-8'))
W = [w for w in W if str(w.get('w', '')).strip()]
_sc = False
for w in W:
    raw = w['w'].strip()
    if raw.startswith('&gt;&gt;') or raw.startswith('>>'): _sc = True
    w['w'] = raw.replace('&gt;', '').replace('&lt;', '').replace('>>', '').strip()
    if w['w']:
        w['sc'] = _sc; _sc = False
W = [w for w in W if w['w']]
for i, w in enumerate(W):
    if 'e' not in w or w['e'] is None:
        w['e'] = min(W[i + 1]['t'], w['t'] + 1.0) if i + 1 < len(W) else w['t'] + 0.4
    if w['e'] <= w['t']:
        w['e'] = w['t'] + 0.12

# ---------------- лексикони за ОСНОВАМИ слів (укр + трохи рос.) ----------------
def ST(txt): return tuple(sorted(set(txt.split())))
CONTRARIAN = ST("""насправді неправд міф брехн ніхто ніде ніколи навпаки неможлив помилк помиля дарма
ілюз парадокс по-справжньому никто нигде никогда наоборот миф лож""")
NEG_CLAIM = ('не можна', 'не буде', 'не працює', 'не вірю', 'не вистачає', 'ніхто не', 'нема кому', 'не так просто', 'такого не буває', 'це не так')
DIRECT = ST("""дивіть уяв знаєте люди слухайте зрозумійте подумайте згадаймо давайте
смотри представ знаете слушайте""")
QUESTION_W = ST("""чому як навіщо хто скільки невже що коли де куди звідки
почему как зачем кто сколько неужели что когда где""")
STORY = ST("""трагікоміч фінал виявля історі легендарн згадаймо раптом буквально позавчора випадок
оказыва истори вдруг однажды""")
HOWTO = ST("""ознак порад правил секрет спосіб перевір насторож головне попаст розпізна""")
OUTCOME = ST("""призвед наслід результат виходить отже підсумок означає по-суті
приведет последств результат значит итог""")
SUPER = ST("""найбільш найгірш найважлив найкращ вперше востаннє єдин унікальн рекорд максимальн мінімальн
самый впервые единствен уникальн""")
VALUE_M = ST("""тому причин означа висновок результат отже виходить факт дан цифр відсот статистик доказ
головне важлив суть принцип закон схем механізм мораль урок фактично реально об'єктивно
потому причин значит вывод факт данные цифры процент суть""")
EMO = {
 'обурення':  ST("""дикіст абсурд безглузд жесть ганьб ужас жах маразм дурниц дуріст сором йо-мойо божевіл
                    сумашедш знущан цинізм нахабств обурлив кошмар дикун"""),
 'гордість':  ST("""геро капелюх неймовірн диво див пиша вража круто шален фантастик неймовір"""),
 'іронія':    ST("""трагікоміч ірон сміх смішн прикольн класн типа типу чарівн казк анекдот"""),
 'тривога':   ST("""тривож небезпе катастроф загроз криз дефіцит черг панік страшн ризик висуш виснаж
                    заблокован блекаут мороз хитат погранич важк складн замерз"""),
 'конфлікт':  ST("""вбил вбит убива полю удар атак ціл тероризм терорист пропаганд ворог війн вторгнен ракет
                    дрон розбомб знищ зруйн обстріл приліт долет б'ють"""),
}
TREND = ST("""україн росі російськ путін зеленськ влад уряд вибор гроші економік війн корупц олігарх протест
майдан реформ кремл санкці інфляц податк бізнес держав політик захід європ нато долар гривн криз бюджет
цін пальн бензин дизел нафт нпз газ мігрант зарплат ринок робот вакансі армі зсу мобіліз бронюван кешбек
укрнафт польщ поляк туреччин сша штат американ китай спирт біоетанол акциз вексел монопол конкуренц парламент
депутат закон суд прокурор поліц тариф пенсі кредит банк енергет електр зерно експорт імпорт логістик порт""")
FILLER = ST("""ну еее ем е-е якби якось коротше короче отже-отже значить блін ото угу ага мм ээ""")
WEAK_START = frozenset("""він вона воно вони це цей ця і а але тому щоб бо ну от значить тоді там який яка яке які
тобто да так он она они это этот и но потому чтобы вот значит тогда""".split())
NUMWORDS = re.compile(r"(\d+[\d.,]*|мільйон\w*|мільярд\w*|тисяч\w*|сотн\w*|відсотк\w*|процент\w*|половин\w*|третин\w*|двічі|втричі|удвічі|у два рази|в два рази)", re.I)
TERM = '.!?…'
_STRIP = '.,!?…:;«»"()—–-'

def norm(t): return t.lower().strip(_STRIP)
def has(tok, stems): return any(tok.startswith(s) for s in stems)
def count(toks, stems): return sum(1 for t in toks if has(t, stems))

# ---------------- речення ----------------
sents, cur = [], []
for i, w in enumerate(W):
    cur.append(i)
    gap = (W[i + 1]['t'] - w['e']) if i + 1 < len(W) else 9
    if w['w'][-1:] in TERM or i == len(W) - 1 or gap > 1.4 or len(cur) >= 40:
        text = ' '.join(W[k]['w'] for k in cur)
        sents.append({'idx': len(sents), 'i0': cur[0], 'i1': cur[-1], 't': W[cur[0]]['t'], 'e': W[cur[-1]]['e'],
                      'text': text, 'nwords': len(cur)})
        cur = []
if not sents:
    print('нема речень'); sys.exit(0)

def toks_of(text):
    return [norm(t) for t in text.split() if norm(t)]

def sent_feats(text):
    toks = toks_of(text); low = text.lower()
    f = dict(ntok=max(1, len(toks)))
    f['contr'] = count(toks, CONTRARIAN) + sum(p in low for p in NEG_CLAIM)
    f['direct'] = count(toks, DIRECT)
    f['q'] = text.count('?') + (1 if toks and toks[0] in QUESTION_W else 0)
    f['story'] = count(toks, STORY)
    f['howto'] = count(toks, HOWTO)
    f['outcome'] = count(toks, OUTCOME)
    f['super'] = count(toks, SUPER)
    f['nums'] = len(NUMWORDS.findall(text))
    f['val'] = count(toks, VALUE_M)
    f['emo'] = {k: count(toks, v) for k, v in EMO.items()}
    f['emo_n'] = sum(f['emo'].values())
    f['tr'] = count(toks, TREND)
    f['fill'] = count(toks, FILLER)
    f['caps'] = max(0, sum(1 for t in text.split()[1:] if t[:1].isupper()))
    return f

def hook_types(f):
    types = []
    if f['contr'] >= 1: types.append('contrarian/bold claim')
    if f['nums'] >= 1: types.append('specific number')
    if f['q'] >= 1: types.append('question')
    if f['direct'] >= 1: types.append('direct address')
    if f['story'] >= 1: types.append('story turn')
    if f['howto'] >= 1: types.append('how-to')
    if f['outcome'] >= 1: types.append('outcome')
    if f['super'] >= 1: types.append('superlative')
    if f['emo_n'] >= 1: types.append('emotional line')
    return types

def quotability(f, nwords, text):
    q = 3 * f['contr'] + 2 * min(f['nums'], 3) + 3 * f['emo_n'] + 2 * f['super'] + 1.5 * f['q'] + 1.5 * f['direct'] + 2 * f['story'] + 3 * f['howto'] + 1.0 * f['outcome']
    if nwords > 22: q -= 3
    if nwords < 4: q -= 5
    q -= 1.0 * f['fill']
    ws = text.split(); k = 0
    while k < min(3, len(ws) - 1) and (norm(ws[k]) in WEAK_START or norm(ws[k]) in FILLER or len(norm(ws[k])) <= 1): k += 1
    fw = ws[k] if k < len(ws) else ''
    if norm(fw) in WEAK_START or fw[:1].isdigit(): q -= 3
    return q

for s in sents:
    s['f'] = sent_feats(s['text'])
    s['q'] = quotability(s['f'], s['nwords'], s['text'])
    s['types'] = hook_types(s['f'])
    s['tr_set'] = {tk for tk in toks_of(s['text']) if has(tk, TREND)}

KEYS = ('contr', 'direct', 'q', 'story', 'howto', 'outcome', 'super', 'nums', 'val', 'emo_n', 'tr', 'fill', 'caps', 'ntok')
def agg(si, sj):
    F = {k: 0 for k in KEYS}; F['emo'] = {k: 0 for k in EMO}; trs = set()
    for k in range(si, sj + 1):
        f = sents[k]['f']
        for key in KEYS: F[key] += f[key]
        for e in EMO: F['emo'][e] += f['emo'][e]
        trs |= sents[k]['tr_set']
    return F, trs

# ---------------- кандидати навколо якорів ----------------
def clamp(x, lo, hi): return max(lo, min(hi, x))

LEAD = WEAK_START | frozenset(FILLER) | frozenset("""да так от ну а і але тобто короче ось ее еее ем взагалі насправді навіть
дивіться слухайте розумієте знаєте бачите уявіть уявляєте власне принципі суті скажімо умовно наприклад теж також ще вже
тут зараз сьогодні ми я нас нам мені у в з на до якщо коли тому що бо щоб чи то це""".split())
LEAD_PHRASES = ('я думаю', 'я вважаю', 'мені здається', 'на мою думку', 'чесно кажучи', 'скажімо так', 'умовно кажучи',
                'я вам скажу', 'я б сказав', 'я би сказала', 'хочу сказати', 'можна сказати', 'так би мовити', 'у нас', 'в нас')
def lead_trim(si):
    """v2.2: старт = перше змістовне слово (референс: редакція ріже лід-ін, медіана 3 слова, до 8);
    зрізаємо філери/сполучники/хеджі («я думаю, що», «ну дивіться», «у нас») — але лишаємо ≥ 4 слова речення"""
    s0 = sents[si]; k = s0['i0']; n = 0
    if s0['text'].rstrip()[-1:] == '?' or norm(W[k]['w']) in QUESTION_W: return k   # питання лишаємо цілим
    while n < 8 and (s0['i1'] - k) >= 4:
        two = (norm(W[k]['w']) + ' ' + norm(W[k + 1]['w'])) if k + 1 <= s0['i1'] else ''
        if two in LEAD_PHRASES:
            k += 2; n += 2; continue
        if norm(W[k]['w']) in LEAD or len(norm(W[k]['w'])) <= 1:
            k += 1; n += 1; continue
        break
    return k
def build(si, sj, anchor):
    k0 = lead_trim(si)
    words = [W[i]['w'] for i in range(k0, sents[sj]['i1'] + 1)]
    text = ' '.join(words)
    return {'si': si, 'sj': sj, 'k0': k0, 't': W[k0]['t'], 'e': sents[sj]['e'], 'dur': sents[sj]['e'] - W[k0]['t'],
            'text': text, 'nwords': len(words), 'anchor': anchor}

def score(c):
    S0 = sents[c['si']]
    F, tr_terms = agg(c['si'], c['sj'])
    ws0 = c['text'].split(); first_tok = norm(ws0[0]) if ws0 else ''
    head = ' '.join(ws0[:12]); fh = sent_feats(head)
    A = sents[c['anchor']]
    # --- HOOK: зона перших 12 слів + якірне речення (якщо воно на старті або в перші 8 с)
    h = 5
    h += min(14, 8 * (1 if fh['contr'] else 0) + 6 * (1 if fh['q'] else 0) + 6 * min(fh['nums'], 1)
             + 4 * (1 if fh['direct'] else 0) + 5 * (1 if fh['story'] else 0) + 4 * (1 if fh['emo_n'] else 0)
             + 3 * (1 if fh['super'] else 0) + 3 * (1 if fh['howto'] else 0))
    anchor_delay = A['t'] - c['t']
    # v2.2: найцитованіший рядок може бути будь-де (референс: медіана позиції 0.3, лише 18 % — на старті);
    # він дає бал як «панчлайн є», а не як «cold-open»
    h += min(7, max(0, A['q'] * 0.7))
    # тест предмета (selection-standard v2.1 «тест контексту»): у перших 6 словах — змістовне слово, число або питання
    subj = any((len(t) >= 5 and t not in LEAD and t not in FILLER) or t[:1].isdigit() or t in QUESTION_W or has(t, TREND)
               for t in [norm(x) for x in ws0[:6]])
    h += 4 if subj else -6
    if first_tok in WEAK_START: h -= 8
    fw = ws0[0] if ws0 else ''
    if S0['nwords'] < 4 or fw[:1].isdigit(): h -= 6   # уламок / старт із числа
    if S0['nwords'] > 28: h -= 3
    hook = clamp(h, 0, 25)
    # --- FLOW
    fl = 13
    fl += 5 if c['text'].rstrip()[-1:] in TERM else -6
    fl -= 2.2 * F['fill'] / math.sqrt(F['ntok'])
    if first_tok in WEAK_START: fl -= 4
    dd = c['dur']
    fl += 4 * math.exp(-((dd - a.peak) ** 2) / (2 * 20 ** 2)) - 1
    if dd > a.softmax: fl -= min(8, 0.5 * (dd - a.softmax))      # референс: 40–65 с ядро, довше — лише повна історія
    if dd < a.min: fl -= 6
    nsc = sum(1 for i in range(c['k0'], sents[c['sj']]['i1'] + 1) if W[i].get('sc'))
    if nsc >= 2: fl -= 2 * (nsc - 1)                              # референс: один спікер; ведучий — лише сетап-питання
    if c['nwords'] < 30: fl -= 6
    wps = c['nwords'] / max(1, dd)
    if 2.0 <= wps <= 3.6: fl += 2
    elif wps < 1.5: fl -= 3
    # завершеність: останнє речення не має бути питанням ведучого (обрив на питанні)
    last = sents[c['sj']]
    if last['text'].rstrip()[-1:] == '?' and last['nwords'] > 6: fl -= 5
    flow = clamp(fl, 0, 25)
    # --- VALUE (факт + інсайт + емоція), щільність на 100 слів
    per100 = 100.0 / max(30, c['nwords'])
    v = 4 + min(6, 1.2 * F['val'] * per100) + min(7, 1.6 * F['nums'] * per100) + min(3, 0.8 * F['caps'] * per100) \
        + min(8, 2.6 * F['emo_n'] * per100) + min(2, 1.0 * F['outcome'] * per100)
    if F['nums'] >= 2 and F['emo_n'] >= 1: v += 2
    value = clamp(v, 0, 25)
    # --- TREND
    t = 5 + min(12, 2.6 * F['tr'] * per100) + min(4, 1.6 * F['emo']['конфлікт'] * per100) + (3 if len(tr_terms) >= 5 else 0) + (1 if len(tr_terms) >= 2 else 0)
    trend = clamp(t, 0, 25)
    total = hook + flow + value + trend
    emo_cls = max(F['emo'].items(), key=lambda kv: kv[1])
    restructure = None   # v2.1: cold-open не робимо; найцитованіший рядок лишається на своєму місці (див. anchor_t)
    return dict(total=round(total), hook=round(hook), flow=round(flow), value=round(value), trend=round(trend),
                types=(A['types'] or hook_types(fh)), emotion=(emo_cls[0] if emo_cls[1] > 0 else '—'),
                anchor_line=A['text'], anchor_t=round(A['t'], 2), anchor_q=round(A['q'], 1), wps=round(wps, 2),
                restructure=restructure)

# (a) якірні вікна: старт = цитований рядок (або коротке речення-сетап перед ним)
anchors = [s for s in sents if s['q'] >= 3.0 and s['nwords'] >= 4]
seen = set(); cands = []
def add(si, sj, ai):
    key = (si, sj)
    if key in seen or sents[si]['t'] < a.teaser: return
    seen.add(key); c = build(si, sj, ai); c.update(score(c)); cands.append(c)
def ends_from(si, ai):
    for sj in range(max(si, ai), len(sents)):
        dur = sents[sj]['e'] - sents[si]['t']
        if dur > a.max + 10: break
        if dur < a.min - 4: continue
        if sents[sj]['text'].rstrip()[-1:] not in TERM: continue
        if sj > si and (sents[sj]['t'] - sents[sj - 1]['e']) > 4.0: break   # велика пауза = кінець теми
        yield sj
for A in anchors:
    ai = A['idx']; starts = [ai]
    if ai > 0 and sents[ai - 1]['nwords'] <= 14 and (A['t'] - sents[ai - 1]['e']) < 3.0:
        starts.append(ai - 1)
    for si in starts:
        for sj in ends_from(si, ai): add(si, sj, ai)
# (b) повний перебір (як у v1): будь-яке речення-старт, якір = найцитованіше речення всередині вікна
for si in range(len(sents)):
    if sents[si]['nwords'] < 4: continue
    for sj in ends_from(si, si):
        ai = max(range(si, sj + 1), key=lambda k: sents[k]['q'])
        add(si, sj, ai)
if not cands:
    print('нема кандидатів'); sys.exit(0)

cands.sort(key=lambda c: (-c['total'], c['t']))
chosen = []
def overlap(x, y): return max(0, min(x['e'], y['e']) - max(x['t'], y['t']))
for c in cands:
    if all(overlap(c, s) / min(c['dur'], s['dur']) < 0.35 for s in chosen):
        chosen.append(c)
    if len(chosen) >= a.top: break

def mmss(s): return f'{int(s // 60):02d}:{int(s % 60):02d}'
print(f'\n=== ТОП-{len(chosen)} КАНДИДАТІВ (рубрика v2.2: предмет → історія → панчлайн) ===\n')
for r, c in enumerate(chosen, 1):
    print(f"{r:2d}. {c['total']:3d}/100  [{mmss(c['t'])}–{mmss(c['e'])}] {c['dur']:.0f}с  H{c['hook']} F{c['flow']} V{c['value']} T{c['trend']}  "
          f"тип: {', '.join(c['types']) or '—'} | емоція: {c['emotion']} | {c['wps']} сл/с")
    print(f"    якір @{mmss(c['anchor_t'])} (q={c['anchor_q']}): «{' '.join(c['anchor_line'].split()[:16])}…»")
    print(f"    старт: «{' '.join(c['text'].split()[:12])}…»")
    if c['restructure']: print(f"    ⟵ RESTRUCTURE: cold-open з {mmss(c['restructure']['t'])} (q={c['restructure']['q']})")
    print()
keep = ('total', 'hook', 'flow', 'value', 'trend', 't', 'e', 'dur', 'nwords', 'types', 'emotion', 'anchor_line', 'anchor_t', 'anchor_q', 'restructure', 'wps', 'text')
with open(a.out, 'w', encoding='utf-8') as fp:
    json.dump([{k: c[k] for k in keep} for c in chosen], fp, ensure_ascii=False, indent=1)
print('→ збережено', a.out)
if a.md:
    L = ['| # | Бал | H | F | V | T | Межі | Довж | Тип хука | Емоція | Якір (хук) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r, c in enumerate(chosen, 1):
        L.append(f"| {r} | **{c['total']}** | {c['hook']} | {c['flow']} | {c['value']} | {c['trend']} | {mmss(c['t'])}–{mmss(c['e'])} | {c['dur']:.0f} с | {', '.join(c['types']) or '—'} | {c['emotion']} | {mmss(c['anchor_t'])} «{' '.join(c['anchor_line'].split()[:12])}…» |")
    open(a.md, 'w', encoding='utf-8').write('\n'.join(L))
    print('→ збережено', a.md)
