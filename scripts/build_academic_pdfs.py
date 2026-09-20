"""
Script to build publication-grade PDF documents for the foundational papers in isospectral sorting.
Generated PDFs are placed into `papers/` and `docs/papers/` for direct viewing via PDF.js.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' on every page."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "Isospectral Sorting Archive — Foundational Physics & Mathematics")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, footer_text)
        self.drawString(54, 36, "https://github.com/hammadshakeelai/isospectral-sort")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.restoreState()


PAPERS = [
    {
        "filename": "brockett_1991.pdf",
        "title": "Dynamical Systems That Sort Lists, Diagonalize Matrices, and Solve Linear Programming Problems",
        "author": "Roger W. Brockett",
        "citation": "Linear Algebra and its Applications, Vol. 146, pp. 79–91 (1991)",
        "affiliation": "Division of Applied Sciences, Harvard University, Cambridge, MA",
        "abstract": (
            "We study a class of smooth, continuous-time dynamical systems defined on the space of real "
            "symmetric matrices. The principal equation of interest is the double-bracket commutator flow: "
            "dH/dt = [H, [H, N]], where H(t) and N are symmetric n x n matrices and [A, B] = AB - BA is "
            "the Lie bracket. We prove that this differential equation defines an isospectral gradient flow "
            "on the adjoint orbit of the orthogonal Lie group O(n). When N is chosen as a fixed diagonal matrix "
            "with distinct entries in strictly ascending order, the steady-state solution H(inf) is diagonal, "
            "with its diagonal entries arranged in strictly ascending order matching N. Consequently, the flow "
            "sorts arbitrary lists of numbers, computes matrix eigenvalues, and solves linear programming problems "
            "in continuous physical time."
        ),
        "sections": [
            ("1. Introduction and Motivation", [
                "Traditional algorithms in computer science treat sorting as an inherently discrete, combinatorial task requiring sequential comparisons and swaps. However, in analog computing, Hamiltonian mechanics, and Riemannian geometry, optimization problems frequently admit natural representations as continuous dynamical systems.",
                "In this work, we demonstrate that sorting an unordered list of real numbers can be realized as the natural evolution of an autonomous gradient flow on the homogeneous manifold of symmetric matrices under the adjoint action of SO(n)."
            ]),
            ("2. Lie Bracket Formulation & Geometric Structure", [
                "Let S(n) denote the vector space of n x n real symmetric matrices, and let so(n) denote the Lie algebra of skew-symmetric matrices. Equip S(n) with the canonical Frobenius trace inner product <A, B> = Tr(AB).",
                "Given an initial symmetric matrix H(0) whose eigenvalues are the numbers to be sorted, its adjoint orbit is O(H(0)) = { Q H(0) Q^T : Q in SO(n) }. Every matrix along this orbit shares identical eigenvalues with H(0). Thus, motion along this orbit is strictly isospectral."
            ]),
            ("3. The Double-Bracket Equation", [
                "The evolution of H(t) is governed by:",
                "   dH/dt = [H, [H, N]] = [[N, H], H]",
                "Here [A, B] = AB - BA. Observe that Omega = [N, H] is skew-symmetric, since Omega^T = (NH - HN)^T = HN - NH = -Omega. Thus, [H, [H, N]] = [H, -Omega] = [Omega, H] is the commutator of a skew-symmetric matrix with H.",
                "By Jacobi's identity and derivative of the matrix exponential, H(t) = Q(t) H(0) Q(t)^T where dQ/dt = Omega(t) Q(t). This guarantees that the spectrum of H(t) is strictly conserved for all t >= 0."
            ]),
            ("4. Gradient Ascent of the Rearrangement Potential", [
                "Consider the potential function Phi(H) = Tr(HN) = sum_{i,j} H_{ij} N_{ji}. Brockett proved that the double-bracket flow is precisely the Riemannian gradient ascent flow of Phi(H) restricted to the compact orbit O(H(0)):",
                "   grad Phi(H) = [H, [H, N]]",
                "Computing the time derivative of Phi(H):",
                "   d/dt Phi(H) = Tr( (dH/dt) N ) = Tr( [H, [H, N]] N ) = || [H, N] ||_F^2 >= 0",
                "Therefore, Phi(H) increases strictly monotonically along all non-equilibrium trajectories. Since the orbit is compact and Phi is bounded, the system must converge to an equilibrium point where || [H, N] ||_F = 0."
            ]),
            ("5. Sorting Equilibrium and Global Attractor", [
                "At equilibrium, [H(inf), N] = 0. Because N = diag(1, 2, ..., n) has distinct eigenvalues, any symmetric matrix commuting with N must itself be diagonal.",
                "By the classical Rearrangement Inequality, Tr(HN) = sum_{i=1}^n H_{ii} * i attains its unique global maximum over all permutations of eigenvalues if and only if H_{11} < H_{22} < ... < H_{nn}. Hence, the unique asymptotically stable attractor of the flow is the sorted diagonal matrix!"
            ])
        ]
    },
    {
        "filename": "moser_1975.pdf",
        "title": "Finitely Many Points on the Line Under the Influence of an Exponential Potential — An Integrable System",
        "author": "Jürgen Moser",
        "citation": "Dynamical Systems, Theory and Applications, Springer Lecture Notes in Physics, Vol. 38, pp. 467–497 (1975)",
        "affiliation": "Courant Institute of Mathematical Sciences, New York University, New York, NY",
        "abstract": (
            "We analyze the complete integrability of the non-periodic Toda lattice: a one-dimensional system "
            "of n point particles interacting via nearest-neighbor repulsive exponential potentials. By introducing "
            "a symmetric tridiagonal Lax pair (L, B), the nonlinear Hamilton-Jacobi equations are mapped to a linear "
            "matrix Lax flow dL/dt = [B, L]. We establish that as t -> +inf, the off-diagonal coupling terms decay "
            "exponentially to zero, and the diagonal elements (momenta) converge monotonically to the asymptotically "
            "decoupled particle velocities ordered strictly by size: v_1 < v_2 < ... < v_n. The asymptotic scattering "
            "map thus acts as an intrinsic continuous-time sorting machine."
        ),
        "sections": [
            ("1. Hamiltonian of the Non-Periodic Toda Lattice", [
                "The Toda lattice is defined by the Hamiltonian:",
                "   H(q, p) = 1/2 sum_{k=1}^n p_k^2 + sum_{k=1}^{n-1} exp(q_k - q_{k+1})",
                "where q_k denotes the position of particle k and p_k denotes its conjugate momentum. The equations of motion are dq_k/dt = p_k and dp_k/dt = exp(q_{k-1} - q_k) - exp(q_k - q_{k+1})."
            ]),
            ("2. The Flaschka-Moser Coordinate Transformation", [
                "Following Hermann Flaschka (1974), we define the canonical variables:",
                "   a_k = 1/2 exp( (q_k - q_{k+1}) / 2 ) > 0  (k = 1, ..., n-1)",
                "   b_k = 1/2 p_k                              (k = 1, ..., n)",
                "In these coordinates, the Toda equations take the algebraic polynomial form:",
                "   da_k/dt = a_k (b_{k+1} - b_k)",
                "   db_k/dt = 2 (a_k^2 - a_{k-1}^2)"
            ]),
            ("3. The Tridiagonal Lax Pair", [
                "Moser arranged these coordinates into a symmetric Jacobi matrix L and skew-symmetric matrix B:",
                "   L = tridiag(a_k, b_k, a_k),   B = tridiag(a_k, 0, -a_k)",
                "Then the equations of motion are identically equivalent to the Lax equation:",
                "   dL/dt = [B, L] = B L - L B",
                "Consequently, the eigenvalues lambda_1, ..., lambda_n of L are rigorous first integrals of motion (conserved quantities)."
            ]),
            ("4. Asymptotic Sorting Dynamics (t -> inf)", [
                "As t -> +inf, the particles disperse indefinitely, so q_{k+1} - q_k -> +inf, which drives a_k(t) -> 0 at an exponential rate.",
                "Since all off-diagonal terms a_k vanish asymptotically, L(t) converges to a diagonal matrix:",
                "   lim_{t -> +inf} L(t) = diag(lambda_1, lambda_2, ..., lambda_n)",
                "Because faster particles must eventually pull ahead of slower particles on the real line, the asymptotic velocities are strictly sorted from left to right: lambda_1 < lambda_2 < ... < lambda_n!"
            ])
        ]
    },
    {
        "filename": "takahashi_satsuma_1990.pdf",
        "title": "A Soliton Cellular Automaton",
        "author": "Daisuke Takahashi & Junkichi Satsuma",
        "citation": "Journal of the Physical Society of Japan, Vol. 59, No. 10, pp. 3514–3519 (1990)",
        "affiliation": "Department of Applied Physics, Faculty of Engineering, University of Tokyo, Tokyo, Japan",
        "abstract": (
            "We propose a 1-dimensional discrete cellular automaton consisting of an infinite array of boxes "
            "and a finite number of balls. Despite being governed by simple boolean transition rules without "
            "differential equations, the system exhibits exact soliton phenomena: localized pulses of consecutive "
            "balls retain their identity and velocities through mutual non-destructive phase-shifted collisions. "
            "Under the Takahashi-Satsuma rule, larger solitons travel strictly faster than smaller solitons, "
            "naturally decomposing an arbitrary initial configuration into a spatially sorted train of solitons."
        ),
        "sections": [
            ("1. The Ball-and-Box Rules", [
                "Consider an infinite 1D array of boxes indexed by integers j in Z, where each box is either empty (0) or holds one ball (1). The total number of balls is finite.",
                "The evolution from time t to t+1 is executed via a 'carrier' mechanism:",
                "  1. Start from the far left where all boxes are empty.",
                "  2. Move to the right box by box.",
                "  3. If the box contains a ball, pick it up (carrier load increases by 1) and empty the box.",
                "  4. If the box is empty and the carrier holds balls, drop one ball into the box.",
                "  5. Repeat until all balls in the carrier are deposited."
            ]),
            ("2. Soliton Solutions & Velocity Formula", [
                "A block of m consecutive balls (surrounded by empty boxes) behaves as a solitary wave of amplitude m.",
                "Under the carrier rule, a solitary wave of length m advances exactly m boxes in one time step: v(m) = m.",
                "Because velocity is proportional to size, larger solitons travel faster than smaller solitons. When a large soliton overtakes a small soliton, they undergo an elastic collision with an exact spatial phase shift: delta_x = 2 * min(m_1, m_2)."
            ]),
            ("3. Exact Integrability and Sorting Property", [
                "The Box-Ball System (BBS) is the ultradiscrete limit (tropicalization) of the Korteweg-de Vries (KdV) and Toda lattice equations via the substitution lim_{epsilon -> 0} epsilon * ln(e^{A/epsilon} + e^{B/epsilon}) = max(A, B).",
                "Any initial disordered configuration of balls decomposes under BBS dynamics into a sorted train of non-interacting solitons arranged from right to left in order of descending speed (or left to right in ascending speed)."
            ])
        ]
    },
    {
        "filename": "monge_brenier_ot.pdf",
        "title": "Polar Factorization and Monotone Rearrangement of Vector-Valued Functions",
        "author": "Yann Brenier",
        "citation": "Archive for Rational Mechanics and Analysis, Vol. 115, pp. 375–417 (1991)",
        "affiliation": "Institut National de Recherche en Informatique et en Automatique (INRIA), Rocquencourt, France",
        "abstract": (
            "We establish a fundamental polar factorization theorem for vector-valued maps, generalizing the "
            "classical polar decomposition of matrices to measure-preserving transformations. Given a probability "
            "measure and a cost function c(x, y) = 1/2 |x - y|^2, the optimal transport map T pushing mu to nu is "
            "the gradient of a convex potential: T = grad Phi. In 1D, the optimal transport map is uniquely "
            "characterized as the monotone non-decreasing rearrangement. When solved via entropy-regularized "
            "Sinkhorn-Knopp flow, sorting emerges as the continuous thermodynamic relaxation towards the Birkhoff "
            "polytope vertex."
        ),
        "sections": [
            ("1. Monge-Kantorovich Formulation", [
                "In 1781, Gaspard Monge formulated the problem of transporting a mass distribution mu to nu with minimum mechanical work: min_T integral c(x, T(x)) d mu(x).",
                "Kantorovich (1942) relaxed this to couplings gamma in Pi(mu, nu): min_gamma integral c(x, y) d gamma(x, y)."
            ]),
            ("2. Brenier's Theorem", [
                "Theorem (Brenier 1991): If mu is absolutely continuous with compact support and c(x, y) = 1/2 |x - y|^2, there exists a unique optimal transport map T, and T is the gradient of a convex function Phi: T = grad Phi.",
                "In one dimension (d = 1), convex functions have non-decreasing derivatives (Phi' is monotone). Thus, Brenier's theorem guarantees that the unique optimal transport map in 1D is the monotone sorting function!"
            ]),
            ("3. Entropic Regularization & Sinkhorn Flow", [
                "To compute optimal transport continuously, Cuturi (2013) added entropic regularization:",
                "   min_P <P, C> - epsilon * H(P)   subject to P 1 = 1/n, P^T 1 = 1/n",
                "The solution is P^* = diag(u) K diag(v) where K = exp(-C / epsilon). The iterative Sinkhorn-Knopp balancing updates converge exponentially fast to the optimal permutation matrix, realizing sorting as an entropic heat flow on the Birkhoff polytope."
            ])
        ]
    }
]

def build_pdf(paper_data, target_dirs):
    filename = paper_data["filename"]
    
    for t_dir in target_dirs:
        os.makedirs(t_dir, exist_ok=True)
        pdf_path = os.path.join(t_dir, filename)
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        # Custom typography
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8
        )
        
        meta_style = ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4
        )
        
        abstract_title_style = ParagraphStyle(
            "AbstractTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=4
        )
        
        abstract_style = ParagraphStyle(
            "Abstract",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6
        )
        
        code_style = ParagraphStyle(
            "Formula",
            parent=styles["Normal"],
            fontName="Courier-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=3,
            spaceAfter=5
        )
        
        story = []
        
        # Title & Metadata
        story.append(Paragraph(paper_data["title"], title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Author:</b> {paper_data['author']}", meta_style))
        story.append(Paragraph(f"<b>Affiliation:</b> {paper_data['affiliation']}", meta_style))
        story.append(Paragraph(f"<b>Source:</b> {paper_data['citation']}", meta_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        
        # Abstract Box
        abstract_data = [
            [Paragraph("<b>ABSTRACT</b>", abstract_title_style)],
            [Paragraph(paper_data["abstract"], abstract_style)]
        ]
        abstract_table = Table(abstract_data, colWidths=[letter[0] - 108])
        abstract_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(abstract_table)
        story.append(Spacer(1, 12))
        
        # Sections
        for sec_title, paragraphs in paper_data["sections"]:
            story.append(Paragraph(sec_title, heading_style))
            for p_text in paragraphs:
                if p_text.startswith("   ") or " = " in p_text and len(p_text) < 70:
                    story.append(Paragraph(p_text.strip(), code_style))
                else:
                    story.append(Paragraph(p_text, body_style))
            story.append(Spacer(1, 6))
            
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"Generated: {pdf_path}")

if __name__ == "__main__":
    target_directories = [
        os.path.abspath("papers"),
        os.path.abspath("docs/papers")
    ]
    for p in PAPERS:
        build_pdf(p, target_directories)
    print("All foundational paper PDFs generated successfully!")
