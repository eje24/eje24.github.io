---
layout: post
title: Complex Analysis - Part 1
author: Ezra
katex: False
---

These exercises are taken from Stein and Shakarchi's _Complex Analysis_, which I've recently started working through with a friend. Solutions and mistakes are my own.

**Problem 1.6** Let $\Omega$ be an opet set in $\cc$ and $z \in \Omega $. The __connected companent__ (or simply the __component__) of $\Omega$ containing $z$ is the set $\mathcal{C}_z$ of all points $w$ in $\Omega$ that can be joined to $z$ by a curve entirely contained in $\Omega$. 


**(a)** Check that $\mathcal{C}_z$ is open and connected. Then, show that $w\in \mathcal{C}_z$ defines an equivalence relation. 

Let $w\in \ccc_z$ be arbitrary. Then, as $w\in \Omega$, there is some open disk $D_w$ centered at $w$ contained in $\Omega$. Since $w\in \ccc_z$, there is a curve contained in $\Omega$ connecting $z$ to $w$, and a similar curve connecting $z$ to each $w' \in D_w$. It follows that there is a curve connecting $z$ to all $w'\in D_w$, and so all such $w'$ are contained in $\ccc_z$, implying that $D_w \in \ccc_z$, and by extension, that $\ccc_z$ is open. It is clear that $\ccc_z$ is connected, and that if $w\in \mathcal{C}_z$, then $z\in \ccc_w$, and so we are done.

Thus, $\Omega$ is the disjoint union of its connected components.

**(b)** Show that $\Omega$ can have only countably many distinct connected components.

From the previous part, the connected components of $\Omega$ are open. Thus, for every $z\in \Omega$, $\ccc_z$ contains some element whose real and imaginary parts are both rational. Since $\qq$ is countable, so too is the number of connected components of $\Omega$.


**(c)** Prove that if $\Omega$ is the complement of a compact set, then $\Omega$ has only one unbounded component.



moo



moo


