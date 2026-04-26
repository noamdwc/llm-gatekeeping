commit	score	adv_f1	benign_f1	fpr	judge_rate	status	description
3c3a19f	0.8879	0.9672	0.8308	0.1562	0.0100	keep	baseline
223a6ea	0.8796	0.9674	0.8254	0.1875	0.0000	discard	disable judge (threshold 0.5)
e31d574	0.8817	0.9641	0.8182	0.1562	0.0350	discard	benign-first classifier prompt plus hard benign examples
48ee816	0.8695	0.9578	0.7941	0.1562	0.0150	discard	tighten adversarial evidence to cut fpr
d6314c8	-1.0000	0.8014	0.4630	0.2188	0.1300	discard	disable few-shot exemplars
c454063	-1.0000	0.7643	0.4500	0.1562	0.8250	discard	widen few-shot confidence anchors
37cdfe0	0.9024	0.9701	0.8485	0.1250	0.0250	keep	adjust few-shot confidence to 95 84
c64e082	0.8817	0.9641	0.8182	0.1562	0.0100	discard	raise benign confidence to 96
b32eba0	0.8796	0.9674	0.8254	0.1875	0.0100	discard	lower attack confidence to 83
ccb005f	0.8732	0.9643	0.8125	0.1875	0.0100	discard	raise attack confidence to 85
fcd3ce2	0.8879	0.9672	0.8308	0.1562	0.0250	discard	lower benign confidence to 94
9ac6638	0.8839	0.9607	0.8116	0.1250	0.0650	discard	add hard benign exemplars
2c1db08	0.9024	0.9701	0.8485	0.1250	0.0250	discard	broaden benign task patterns
380ea2b	0.8287	0.9458	0.7353	0.2188	0.1400	discard	reduce nlp exemplars to 4
df37e87	0.8732	0.9643	0.8125	0.1875	0.0300	discard	reduce unicode exemplars to 1
f5df57a	0.9024	0.9701	0.8485	0.1250	0.0250	discard	switch to dynamic few-shot
23366d2	0.8732	0.9643	0.8125	0.1875	0.0300	discard	shorten unicode evidence snippets
b314da4	0.9024	0.9701	0.8485	0.1250	0.0250	discard	dynamic few-shot with k 3
7073780	0.9024	0.9701	0.8485	0.1250	0.0250	discard	lower judge threshold to 0.79
4ae94ab	0.8755	0.9610	0.8060	0.1562	0.0150	discard	make benign few-shot reason explicit
63f8adb	0.0000	0.0000	0.0000	0.0000	0.0000	crash	make attack few-shot reasons explicit
8b1aa7e	0.8755	0.9610	0.8060	0.1562	0.0050	discard	clarify productivity benign boundary
0290a37	0.8781	0.9474	0.7792	0.0625	0.0250	discard	increase unicode exemplars
808622a	0.8817	0.9641	0.8182	0.1562	0.0000	discard	lengthen unicode evidence span
86caba7	0.9028	0.9568	0.8158	0.0312	0.0050	keep	increase nlp exemplars
62c5aaf	0.8755	0.9610	0.8060	0.1562	0.0050	discard	increase nlp exemplars again
0d3238d	0.9028	0.9568	0.8158	0.0312	0.0050	discard	raise judge threshold slightly
398800f	0.8270	0.9390	0.7222	0.1875	0.0750	discard	increase unicode exemplars with nlp6
353efca	0.8922	0.9605	0.8169	0.0938	0.0000	discard	benign conf 94
