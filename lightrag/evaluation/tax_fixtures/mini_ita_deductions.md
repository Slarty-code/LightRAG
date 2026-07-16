# Income Tax Assessment Act 1997 (Synthetic Mini)

Synthetic mini-Act for LightRAG tax retrieval and chunk-integrity evaluation.
Hierarchy and wording are abbreviated for fixtures only; they are not official text.

## Part 1 — Preliminary

This Act provides for assessable income, deductions, and related administration.
Defined terms used later in this Act have the meaning given in this Part unless the contrary intention appears.

The objects of this Act include identifying amounts that form part of assessable income and specifying the circumstances in which a taxpayer may deduct a loss or outgoing. Preliminary notes remind readers that operative rules live in later Parts and Divisions rather than in this introduction.

## Part 2 — Deductions

This Part sets out the general rules for deductions from assessable income.
Unless a contrary intention appears, a deduction under this Part is a general deduction.
Specific deductions in other Divisions of this Act operate in addition to, and do not narrow, the general deduction in Division 8 unless expressly stated.

Readers looking for the operative deduction rules should begin with Division 8 of this Part before consulting administrative notes elsewhere in the Act.

### Division 8 — General deductions

Division 8 states the central positive rule and supporting notes for general deductions.
The Division begins with section 8-1 and continues with section 8-5.
Those sections are the primary anchors for general-deduction questions in this synthetic Act.

Section markers in this Division use the form ``s N-N`` so retrieval and chunk integrity checks can detect boundary breaks when a marker is split across adjacent chunks.

#### s 8-1

General deductions.
You can deduct from your assessable income any loss or outgoing to the extent that it is incurred in gaining or producing your assessable income, or is necessarily incurred in carrying on a business for the purpose of gaining or producing your assessable income.

A loss or outgoing is not deductible under this section to the extent that it is a loss or outgoing of capital, or of a capital nature; is of a private or domestic nature; or is incurred in relation to gaining or producing exempt income or non-assessable non-exempt income.

Worked illustration (synthetic): a freelance designer who buys software solely to complete client invoices incurs an outgoing to the extent that the software is used in gaining assessable income. The same designer cannot rely on this section for the private half of a dual-use device without apportionment.

Padding note A for fixture sizing: keep enough prose under s 8-1 that fixed-token chunking may cut mid-paragraph while heading-aware chunking retains the section marker with its operative text. Additional sentences restate the to the extent apportionment idea without introducing penalty language from later Parts.

Padding note B for fixture sizing: general deduction inquiries should retrieve this section together with the Division heading rather than administrative material that merely mentions the word deduction in another Part.

#### s 8-5

Further notes on general deductions.
A deduction under section 8-1 is the leading case of a general deduction for the purposes of this Act. Later Divisions may allow specific deductions that apply even when section 8-1 would not.

Nothing in this section authorises a double deduction for the same loss or outgoing. Where both a general deduction and a specific deduction could apply, follow the interaction rules stated in those Divisions.

Padding note C for fixture sizing: section 8-5 clarifies that general deduction status attaches primarily through s 8-1 and that specific regimes sit atop that baseline. Keep the marker ``s 8-5`` intact inside a heading-aware chunk so oracle checks can require the expected section anchors.

## Part 3 — Assessable income

Assessable income includes ordinary income and statutory income according to the ordinary concepts of this synthetic Act.
This Part deliberately avoids restating the Division 8 deduction tests. Cross-references to deductions appear only where necessary to distinguish assessable amounts from amounts that may later be deducted under Part 2.

Ordinary income derived from personal services, business receipts, and similar sources is included unless a provision of this Act says otherwise. Statutory income is included only as expressly provided.

## Part 4 — Penalties and offences

This Part deals with administrative penalties and offences for false or misleading statements.
It is not the Deduction Division and does not state when a loss or outgoing is deductible.

Important warning: there is a penalty for false or misleading statements made in connection with a deduction claim that lacks supporting evidence. That single mention of a deduction relates only to the penalty setting and must not be treated as the operative Deduction Division text.

Officers may require a taxpayer to produce records supporting amounts claimed. Failure to comply may attract a separate administrative consequence under this Part. Keep this material isolated from Division 8 when evaluating heading-aware chunking.
