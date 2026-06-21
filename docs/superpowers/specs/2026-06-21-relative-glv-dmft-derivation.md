# Dynamical mean-field theory of the relative (scale-invariant) GLV

Date: 2026-06-21. Working derivation; to be ported to the thesis as a section + appendix once validated.

Model lives in `growing_glv/explore.py`, `phase_space_compute.py`, `phase_diagram.py`. This document derives
its DMFT from scratch, shows it is a small extension of the Aguirre-Lopez heterogeneous DMFT already cited in
`thesis/chapters/dynamical_equations.tex`, and lists what it predicts and where it breaks.

---

## 0. Result in one box

The relative GLV is a **disordered replicator equation**. Its single representative firm obeys

$$
\dot y = y\Big[\,1 - y - \mu M_1(t) - g(t) + \sigma\,\eta(t) + \gamma\sigma^2\!\int_0^t\! G(t,s)\,y(s)\,ds\,\Big],
$$

with the four self-consistency closures

$$
M_1(t)=\mathbb E[y(t)]=1,\quad
\langle\eta(t)\eta(s)\rangle=C(t,s)=\mathbb E[y(t)y(s)],\quad
G(t,s)=\mathbb E\!\left[\tfrac{\delta y(t)}{\delta\zeta(s)}\right],\quad
g(t)=\mathbb E[y(t)F(t)].
$$

This is **exactly** the effective process of `eq:dmft-effective` in the thesis, with carrying capacity
$K=1$, the competition sign on the mean field, and **two additions**:

1. a common drift $-g(t)$, the mean-fitness subtraction of the replicator (the per-capita growth rate of the economy);
2. the normalisation $M_1\equiv1$, which is no longer free but pinned by scale invariance, and whose closure
   $g(t)=\mathbb E[yF]$ determines $g(t)$.

At a fixed point (the relaxed phase) it collapses to the Bunin/Galla GLV self-consistency with $K\to 1-g^\*$
plus one extra scalar equation $M_1=1$ fixing $g^\*$. The relaxed$\leftrightarrow$fluctuating boundary is
$\sigma^2\varphi=(1-\gamma\sigma^2\chi)^2$; for $\gamma=0,\mu=0$ this gives $\sigma_c=\sqrt2$, the line already
drawn by hand in `phase_diagram.py:72`.

---

## 1. Model and conventions

Roy competition sign, matching the code (`phase_space_compute.py:69`):

$$
\dot x_i = x_i\Big[\,1 - \frac{x_i}{m} - \frac{(\alpha x)_i}{m}\,\Big],
\qquad m=\frac1N\sum_j x_j=\frac MN .
\tag{1}
$$

Couplings on a configuration-model graph of mean degree $C$:

$$
\alpha_{ij}=A_{ij}\Big(\frac{\mu}{C}+\frac{\sigma}{\sqrt C}\,z_{ij}\Big),\quad
z_{ij}\sim\mathcal N(0,1),\quad \overline{z_{ij}z_{ji}}=\gamma,\quad \alpha_{ii}=0 .
$$

Both the self-limitation $x_i/m$ and the interaction $(\alpha x)_i/m$ are divided by the population mean $m$.
That is the one change from the absolute GLV of `eq:glv`, and it is decisive: the right-hand side of (1) is
**homogeneous of degree 1** in $x$ (because $m\propto x$), so $F_i:=1-x_i/m-(\alpha x)_i/m$ is degree $0$ and
depends on $x$ only through ratios. The overall scale is a zero mode: it grows (or decays) exponentially and
never blows up in finite time. ($\mu>0$ is net competition here; the cooperative-sign chapter has $\mu\to-\mu$.)

## 2. Reduction to a replicator equation

Define relative abundances $y_i=x_i/m$, so $\langle y\rangle:=\frac1N\sum_i y_i=1$ identically, and $M=Nm$.
The aggregate grows at the population-averaged fitness:

$$
g(t):=\frac{d\ln m}{dt}=\frac{d\ln M}{dt}
=\frac1M\sum_j\dot x_j=\sum_j\frac{x_j}{M}F_j=\frac1N\sum_j y_jF_j=\langle yF\rangle .
$$

Subtracting the scale, $\dfrac{d\ln y_i}{dt}=F_i-g(t)$, i.e.

$$
\boxed{\;\dot y_i=y_i\big(F_i-g(t)\big),\qquad F_i=1-y_i-(\alpha y)_i,\qquad g(t)=\langle yF\rangle.\;}
\tag{2}
$$

Equation (2) is a replicator equation. The subtraction $g(t)$ enforces the normalisation: $\frac{d}{dt}\langle
y\rangle=\langle yF\rangle-g\langle y\rangle=g\,(1-\langle y\rangle)$, so $\langle y\rangle=1$ is invariant. The
economy is $M(t)=M(0)\exp\!\int_0^t g$, and $g^\*>0$ in the stationary state gives a steadily growing economy
with no finite-time singularity. This is the Hofbauer Lotka-Volterra $\leftrightarrow$ replicator
correspondence already noted in the thesis, here applied to the scale-invariant model so that the scale
equation is the physically interesting growth rate rather than a discarded divergence.

## 3. The local field and the dynamical cavity argument

Take the regular graph first ($k_i\equiv C$); the degree-heterogeneous case is Section 7. Split the interaction
on node $i$ into mean and fluctuation:

$$
(\alpha y)_i=\underbrace{\frac{\mu}{C}\sum_{j\in\partial i}y_j}_{\to\ \mu M_1}
+\underbrace{\frac{\sigma}{\sqrt C}\sum_{j\in\partial i}z_{ij}y_j}_{=:h_i(t)} .
$$

In the high-connectivity limit the local neighbourhood average equals the global one, so the mean term is $\mu
M_1$ with $M_1=\langle y\rangle$. The fluctuating field $h_i$ is treated by the dynamical cavity method (Bunin
2017; Barbier et al. 2018), identical in structure to the absolute GLV because $(\alpha y)$ is linear in $y$.

Remove node $i$. The cavity abundances $y_j^{(i)}$ are independent of $\{z_{ij},z_{ji}\}$. Reintroducing $i$
perturbs each neighbour by linear response. The field $i$ exerts on $j$ is $-\,\alpha_{ji}y_i$, whose
fluctuating part is $-\frac{\sigma}{\sqrt C}z_{ji}\,y_i$; with $G_j(t,s)=\delta y_j(t)/\delta\zeta_j(s)$ the
response of $j$ to a source $\zeta_j$ in its bracket,

$$
\delta y_j(t)=-\int_0^t\!ds\,G_j(t,s)\,\frac{\sigma}{\sqrt C}\,z_{ji}\,y_i(s).
$$

Therefore

$$
h_i(t)=\frac{\sigma}{\sqrt C}\sum_{j\in\partial i}z_{ij}\,y_j^{(i)}(t)
\;-\;\frac{\sigma^2}{C}\sum_{j\in\partial i}z_{ij}z_{ji}\!\int_0^t\!ds\,G_j(t,s)\,y_i(s).
$$

The first sum is $C$ independent zero-mean terms; by the CLT it is a Gaussian noise $\xi_i(t)$ with

$$
\langle\xi_i(t)\xi_i(s)\rangle=\frac{\sigma^2}{C}\sum_{j\in\partial i}y_j(t)y_j(s)
=\sigma^2\,C(t,s),\qquad C(t,s):=\langle y(t)y(s)\rangle .
$$

The second sum uses $\overline{z_{ij}z_{ji}}=\gamma$ and $\frac1C\sum_{j\in\partial i}G_j\to G:=\langle G_j\rangle$,
giving the retarded (Onsager) reaction $-\gamma\sigma^2\int_0^t G(t,s)\,y_i(s)\,ds$. Substituting
$(\alpha y)_i=\mu M_1+h_i$ into $F_i=1-y_i-(\alpha y)_i$ and dropping the species index yields the **effective
single-site process**

$$
\boxed{\;\dot y=y\Big[\,1-y-\mu M_1(t)-g(t)+\sigma\,\eta(t)+\gamma\sigma^2\!\int_0^t\!G(t,s)\,y(s)\,ds\,\Big]\;}
\tag{3}
$$

where $\eta=-\xi/\sigma$ is the unit-strength Gaussian noise, $\langle\eta(t)\eta(s)\rangle=C(t,s)$ (its sign is
immaterial). The same retarded sign $+\gamma\sigma^2$ arises in both the cooperative and competition
conventions; (3) is `eq:dmft-effective` with $K=1$, the competition mean-field sign, and the new $-g(t)$. The
generating-functional (MSRJD) route gives the same (3); the cavity version is shown because it is the one the
ecology-DMFT literature the thesis cites uses.

## 4. Self-consistency (general, two-time)

The disorder average has been traded for four functionals of the single-site law of (3):

$$
M_1(t)=\mathbb E[y(t)]=1,\qquad
C(t,s)=\mathbb E[y(t)y(s)],\qquad
G(t,s)=\mathbb E\!\Big[\frac{\delta y(t)}{\delta\zeta(s)}\Big]_{\zeta=0},
$$

$$
g(t)=\mathbb E[y(t)F(t)],\qquad F=1-y-\mu M_1+\sigma\eta+\gamma\sigma^2\!\int_0^t G\,y .
$$

The first is the normalisation. The middle two are the standard correlation and response that close the noise
and memory. The last is the **new closure**: it is forced by consistency, since $\dot M_1=\mathbb
E[yF]-g\,M_1$, and demanding $\dot M_1=0$ at $M_1=1$ gives exactly $g(t)=\mathbb E[yF]$. Crucially $g(t)$ is a
*global* quantity ($\frac1N\sum_j y_jF_j$), so it self-averages: its fluctuations are $O(1/\sqrt N)$ and it is a
deterministic function in the limit, a common drift rather than an extra noise. It is the only structural
addition to the standard GLV self-consistency.

## 5. Fixed point: the relaxed phase

In the single-fixed-point phase the shares relax, $y(t)\to y^\*$, and the statistics become
time-translation-invariant. With $y(t)y(s)\to y^{\*2}$ the noise correlation $C(t,s)\to q:=\mathbb E[y^{\*2}]$
is constant for all $t,s$, so the colored noise freezes to a single static Gaussian $\sigma\eta\to\sigma\sqrt
q\,z$, $z\sim\mathcal N(0,1)$. The memory becomes $\gamma\sigma^2\int_0^t G(t,s)y(s)\,ds\to\gamma\sigma^2\chi\,
y^\*$ with the static susceptibility $\chi=\int_0^\infty G(u)\,du$. Setting $\dot y=0$ in (3):

$$
y^\*(z)=\frac{(1-g^\*)-\mu M_1+\sigma\sqrt q\,z}{\,1-\gamma\sigma^2\chi\,}\quad\text{if positive, else }0 .
\tag{4}
$$

This is the Bunin/Galla survivor law $y^\*=\big(K-\mu M_1+\sigma\sqrt q\,z\big)/(1-\gamma\sigma^2\chi)$ with the
**carrying capacity renormalised by the growth rate, $K\to 1-g^\*$**. Equivalently: at a replicator fixed point
every survivor has $F_i^\*=g^\*$ (fitnesses equalise), i.e. $y_i^\*=(1-g^\*)-(\alpha y^\*)_i$, the LV fixed
point with capacity $1-g^\*$.

Write $v=1-\gamma\sigma^2\chi$, $\Delta=\dfrac{1-g^\*-\mu M_1}{\sigma\sqrt q}$ (so survivors have $z>-\Delta$),
and the Gaussian moments $w_n(\Delta)=\int_{-\Delta}^\infty Dz\,(\Delta+z)^n$, $Dz=e^{-z^2/2}dz/\sqrt{2\pi}$:

$$
w_0=\Phi(\Delta),\quad w_1=\Delta\Phi(\Delta)+\phi(\Delta),\quad w_2=(1+\Delta^2)\Phi(\Delta)+\Delta\phi(\Delta).
$$

The four self-consistency equations, unknowns $(\Delta,q,\chi,g^\*)$ with $M_1\equiv1$:

$$
\begin{aligned}
\text{(survival)}\quad & \varphi=w_0(\Delta),\\
\text{(normalisation, fixes }g^\*)\quad & M_1=\frac{\sigma\sqrt q}{v}\,w_1(\Delta)=1,\\
\text{(mean square)}\quad & q=\Big(\frac{\sigma\sqrt q}{v}\Big)^{\!2}w_2(\Delta)\ \Rightarrow\ v^2=\sigma^2 w_2(\Delta),\\
\text{(response)}\quad & \chi=\frac{\varphi}{v},\qquad v=1-\gamma\sigma^2\chi .
\end{aligned}
\tag{5}
$$

This is precisely the structure of `eq:muc-selfconsistency`, with one extra unknown $g^\*$ and one extra
equation (the normalisation). $g^\*$ is read off the converged solution and is the analog of the thesis shape
scalar $c$: $g^\*=\langle y^\*F^\*\rangle$, the steady growth rate of the economy. $g^\*>0$ defines the growing
region (compare `growth_scan.py` / `g_eff`).

## 6. Stability and the phase boundary

The fixed point is linearly stable until the disorder feedback develops a zero mode. The marginal condition is

$$
\boxed{\;\sigma^2\varphi=(1-\gamma\sigma^2\chi)^2=v^2.\;}
\tag{6}
$$

**Check** ($\gamma=0,\mu=0$): $v=1$, so (6) is $\sigma^2\varphi=1$. The mean-square equation gives
$\sigma^2 w_2=1$, so on the boundary $\varphi=w_2$, i.e. $w_0(\Delta)=w_2(\Delta)$, solved by $\Delta=0$ (then
$\varphi=\tfrac12$, $w_2(0)=\tfrac12$). Hence $\sigma_c^2=1/w_2(0)=2$, **$\sigma_c=\sqrt2$ with half the species
surviving at threshold** — the Galla/Bunin value, and exactly the line guessed in `phase_diagram.py:72`.

At threshold $\Delta_c=0$, so $\mu$ and $g^\*$ drop out of the threshold condition entirely: on the regular
graph **the boundary is $\mu$-independent**, a horizontal line $\sigma_c(\gamma)$ ($=\sqrt2$ at $\gamma=0$).
This is because the mean competition $\mu$ enters as a uniform shift of every fitness, which cancels in the
replicator (Section 2), so it cannot change the relaxed/fluctuating structure — it only moves the $g^\*=0$
growth contour. (Degree heterogeneity breaks this: $\kappa\mu$ varies by node, so $\mu$ re-enters and the
boundary curves.) Below it: a single fixed point, shares relax, the economy grows steadily at $g^\*$ — the green
"relaxed" region of the thesis shape-relaxation figure and the squares in `phase_diagram.png`. Above it:
persistent fluctuations on the simplex (the Opper-Diederich "$1/f$" phase), where the MSB tent and fat tails
live — the circles in `phase_diagram.png`. The DMFT thus predicts the *same* curve the thesis maps
numerically.

## 7. Degree heterogeneity

Following Aguirre-Lopez and the thesis, a node of rescaled degree $\kappa=k/C$ has the effective process

$$
\dot y=y\Big[\,1-y-\kappa\mu M_1-g(t)+\sqrt\kappa\,\sigma\,\eta(t)+\kappa\gamma\sigma^2\!\int_0^t\!G\,y\,\Big],
$$

and the order parameters become averages over the rescaled degree law $\nu(\kappa)$:
$M_1=\int d\kappa\,\nu(\kappa)\,\mathbb E_\kappa[y]=1$, $C(t,s)=\int\nu\,\mathbb E_\kappa[y(t)y(s)]$,
$G=\int\nu\,\mathbb E_\kappa[\delta y/\delta\zeta]$, $g=\int\nu\,\mathbb E_\kappa[yF]$. The fixed-point system is
`eq:muc-selfconsistency` with $K\to1-g^\*$ and the added normalisation. The normalisation is on the flat node
average ($\langle y\rangle=1$ regardless of degree); the degree enters only the fitness.

**Caveat.** `build_alpha` uses a power-law degree with exponent $2.5$, for which $\langle\kappa^2\rangle$
diverges. The disorder terms scale as $\kappa$ and the variance contributions as $\kappa$, so the
$\kappa$-weighted integrals do not converge and the DMFT is ill-posed on that graph. Use the exponential or
regular degree (as the $\mu_c$ chapter already does), or treat hub dominance as a separate, genuinely
non-mean-field story.

## 8. Observables

| DMFT output | How | Numerical counterpart |
|---|---|---|
| growth rate $g^\*(\mu,\sigma)$ | solve (5) | `g_eff`, `growth_scan.py` |
| survival fraction $\varphi(\mu,\sigma)$ | $w_0(\Delta)$ | `surv_frac` |
| size distribution $P(y)$ | pushforward of $\mathcal N(0,1)$ through (4) at FP; stationary marginal of (3) in the fluctuating phase | relative-size histograms |
| phase boundary $\sigma_c$ ($\mu$-independent, $=\sqrt2$ at $\gamma=0$) | (6) + (5) | `phase_diagram.png`, shape-relaxation figure |
| MSB exponent $\beta$ | two-time DMFT (below) | `beta`, `beta_scaling.py` |

**The $\beta$ payoff.** The single-firm log-growth is $r(t)=d\ln y/dt=F(t)-g(t)$, and the MSB law is
$\mathrm{vol}(y)\propto y^{-\beta}$. In the DMFT the noise $\eta$ is a *common* field: there is no genuine
cross-firm correlation by construction. So any $\beta>0$ the DMFT produces comes purely from the single-site
mechanics — the self-limitation $-y$ (larger firms mean-revert harder) and the memory term — not from a
correlation between firms. Computing $\mathrm{vol}(y)$ from the stationary single-site process therefore yields
$\beta(\mu,\sigma)$ **and proves $\beta$ is a mean-field, self-limitation effect rather than evidence for the
size-growing-correlation story.** This is the analytic version of the ablation result ($W=0$ keeps $\beta\sim
0.15$) and answers the thesis's central question head-on. Because $\beta$ is a fluctuation property it lives in
the fluctuating phase only ($y^\*$ is static, no volatility), so it needs the full two-time DMFT — the
numerically integrated $C(t,s),G(t,s)$ (Eissfeller-Opper / Roy 2019 scheme), or the persistent-fluctuation
closure of Opper-Diederich.

## 9. Validity and precedent

- Exact for $N\to\infty$ in the high-connectivity limit (dense, or $1\ll C\ll N$ and locally tree-like) — the
  same regime the thesis already assumes for $\mu_c$.
- The fixed-point system (5)-(6) describes the relaxed (steady-growth) region only. The MSB tent and tails are
  in the fluctuating phase and need the two-time DMFT.
- Power-law degree breaks the $\kappa$-integrals (Section 7).
- The disordered-replicator DMFT is the original setting of Opper-Diederich (PRL 1992, "Phase transition and
  $1/f$ noise in a game dynamical model"); see also Galla (2006) and Sidhom-Galla (PRE 2020). The relative GLV
  is that model with the GLV self-limitation included.

## 10. Next steps

1. ~~Code the (5)-(6) fixed-point solver and check it against simulation.~~ **Done, Section 11.**
2. Two-time DMFT integrator for the fluctuating phase $\Rightarrow$ $\beta(\mu,\sigma)$ and $P(y)$ analytically.
   This is the regime ($\sigma>\sigma_c$) where the MSB tent lives; the static solver cannot reach it.
3. Heterogeneous-degree version (Section 7) on an exponential/regular graph, to compare against the
   power-law production runs (`phase_space_data.npz`).
4. Port to the thesis: a derivation section (Sections 1-6) plus a cavity appendix (Section 3), in house style.

## 11. Validation (2026-06-21)

`growing_glv/dmft_solver.py` solves (5)-(6); 12 analytic-limit tests in `test_dmft_solver.py` pass.
`growing_glv/dmft_validate.py` runs a matched direct simulation (fully-connected Gaussian, $\gamma=0$,
$N=1500$, $\lambda=0$ — the DMFT's exact ensemble) and stores it in `dmft_validation.npz`
(`dmft_validation.png`). Results, no fit:

- **Relaxed phase** ($\sigma<\sqrt2$): mean $|g_{\rm eff}-g^\*|=0.021$, mean $|{\rm surv}-\varphi|=0.010$ over
  $\sigma\in\{0.4,\dots,1.35\}$ at $\mu=0.5$. E.g. $\sigma=1.0$: sim $(-0.199,0.690)$ vs DMFT $(-0.195,0.681)$.
- **Chaos onset at $\sigma_c=\sqrt2$**: the late-time fluctuation amplitude is $\le0.005$ for $\sigma<\sqrt2$
  and jumps to $0.038,0.13,0.21$ at $\sigma=1.6,2.0,2.5$.
- **$\mu$-independence confirmed**: at $\sigma=1.0$, sweeping $\mu=0\to1\to2$ leaves survival fixed
  ($0.688/0.687/0.685$) and shifts $g_{\rm eff}$ by exactly $-\mu$ ($+0.317/-0.682/-1.681$).
- **Fluctuating phase** ($\sigma>\sqrt2$): the static $g^\*,\varphi$ overshoot the simulation (e.g. $\sigma=2.5$:
  $g^\*=4.55$ vs sim $3.95$; $\varphi=0.26$ vs sim $0.09$), as expected since the FP is unstable there — the
  quantitative signal that step 2 (two-time DMFT) is needed.
