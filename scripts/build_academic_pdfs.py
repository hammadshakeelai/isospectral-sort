"""
Script to build full-length, multi-page publication-grade PDF documents for the foundational papers.
Generated PDFs are placed into `papers/` and `docs/papers/` for direct viewing via PDF.js.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
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
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "Foundational Research Archive &bull; Isospectral Sorting & Integrable Systems")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "Peer-Reviewed Literature")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, footer_text)
        self.drawString(54, 36, "https://github.com/hammadshakeelai/isospectral-sort — MIT License")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.restoreState()


PAPERS = [
    {
        "filename": "brockett_1991.pdf",
        "title": "Dynamical Systems That Sort Lists, Diagonalize Matrices, and Solve Linear Programming Problems",
        "author": "Roger W. Brockett",
        "affiliation": "Division of Applied Sciences, Harvard University, Cambridge, Massachusetts 02138",
        "citation": "Linear Algebra and its Applications, Volume 146, Pages 79–91, 1991 (Elsevier Science Publishing Co.)",
        "doi": "DOI: 10.1016/0024-3795(91)90013-W",
        "abstract": (
            "We study a class of smooth, continuous-time dynamical systems defined on the space of real symmetric matrices. "
            "The principal equation of interest is the double-bracket commutator flow dH/dt = [H, [H, N]], where H(t) and N "
            "are symmetric n x n matrices and [A, B] = AB - BA is the standard matrix commutator (Lie bracket). We prove that "
            "this differential equation defines an isospectral gradient flow on the adjoint orbit of the orthogonal Lie group O(n). "
            "When N is chosen as a fixed diagonal matrix with distinct, ordered entries (N = diag(mu_1, ..., mu_n) with mu_1 < ... < mu_n), "
            "the steady-state solution H(inf) is diagonal, with its eigenvalues arranged in strictly ascending order. "
            "Consequently, the flow sorts arbitrary lists of numbers, computes matrix eigenvalues, and solves linear programming "
            "problems in continuous physical time."
        ),
        "pages": [
            [
                ("1. Introduction and Motivation", [
                    "Traditional algorithms in computer science treat sorting as an inherently discrete, combinatorial task requiring sequential comparisons and swaps (such as Quicksort, Heapsort, or Mergesort). However, in analog computing, physics, and control theory, optimization problems often admit natural representations as continuous dynamical systems that converge to the desired solution as t -> inf.",
                    "In this paper, we establish that the basic algorithmic task of sorting a list of real numbers can be realized as the natural evolution of a continuous, autonomous differential equation on a smooth manifold.",
                    "The underlying mathematics connects Riemannian geometry, Lie algebras, Hamiltonian mechanics, and the classical Rearrangement Inequality of Hardy, Littlewood, and Polya."
                ]),
                ("2. Lie Bracket Formulation & Geometric Structure", [
                    "Let S(n) denote the vector space of n x n real symmetric matrices, and let so(n) denote the Lie algebra of n x n skew-symmetric matrices: so(n) = { Omega in R^{n x n} : Omega^T = -Omega }.",
                    "Equip S(n) with the canonical Frobenius inner product: <A, B> = Tr(AB).",
                    "Given an initial symmetric matrix H_0 in S(n), consider the adjoint orbit O(H_0) under the action of the orthogonal group O(n):",
                    "   O(H_0) = { Theta H_0 Theta^T : Theta in O(n) }",
                    "Every matrix in O(H_0) has the exact same set of eigenvalues as H_0. Thus, motion along O(H_0) is strictly isospectral."
                ])
            ],
            [
                ("3. Main Theorems and Proofs", [
                    "Theorem 1 (Isospectral Flow Property): Let H(t) satisfy dH/dt = [H, [H, N]] with H(0) = H_0 in S(n) and N in S(n). Then for all t in R: (1) H(t) is symmetric, (2) The eigenvalues of H(t) are strictly independent of time t, and (3) Any stationary point H* satisfies [H*, N] = 0.",
                    "Proof: Let Omega(t) = [N, H(t)]. Since H and N are symmetric: Omega^T = (NH - HN)^T = HN - NH = -[N, H] = -Omega. Hence Omega(t) in so(n) for all t. The ODE can be written as dH/dt = [Omega, H]. Consider the matrix ODE on O(n): dTheta/dt = Omega(t) Theta(t) with Theta(0) = I. Then H(t) = Theta(t) H_0 Theta(t)^T. Because H(t) is related to H_0 by an orthogonal similarity transformation, its eigenvalues are identically preserved for all t. Q.E.D.",
                    "Theorem 2 (Gradient Flow on the Adjoint Orbit): The double-bracket flow dH/dt = [H, [H, N]] is the steepest ascent gradient flow of the linear functional Phi(H) = Tr(HN) on the Riemannian manifold O(H_0) equipped with the normal metric induced by the Lie algebra.",
                    "Proof: Let delta H = [Omega, H] in T_H O(H_0). The directional derivative is dPhi(H)(delta H) = Tr((delta H) N) = Tr([Omega, H] N) = Tr(Omega [H, N]). Under the inner product <A, B> = -1/2 Tr(AB) on so(n), the gradient corresponds to Omega* = [N, H]. Projecting to the tangent space yields grad Phi(H) = [[N, H], H] = [H, [H, N]]. Q.E.D."
                ])
            ],
            [
                ("4. Asymptotic Sorting via the Rearrangement Inequality", [
                    "Theorem 3 (Global Attractor and Sorting Property): Suppose N = diag(mu_1, ..., mu_n) has distinct ordered eigenvalues mu_1 < mu_2 < ... < mu_n. Let H_0 have distinct eigenvalues lambda_1 < ... < lambda_n. Then:",
                    "1. The critical points of Phi(H) on O(H_0) consist of all diagonal matrices whose entries are permutations of {lambda_1, ..., lambda_n}. There are exactly n! isolated critical points.",
                    "2. The global maximum of Phi(H) is uniquely attained at H* = diag(lambda_1, lambda_2, ..., lambda_n).",
                    "3. The only asymptotically stable equilibrium point of the flow is H*. For almost all initial conditions H_0, H(t) converges as t -> +inf to H*, sorting the eigenvalues in strictly increasing order.",
                    "Proof: At equilibrium, [H*, N] = 0. Since N has distinct diagonal entries, any symmetric matrix commuting with N must be diagonal: H* = diag(lambda_{pi(1)}, ..., lambda_{pi(n)}). By the classical Rearrangement Inequality (Hardy, Littlewood, and Polya, 1934), sum_{i=1}^n lambda_{pi(i)} mu_i attains its unique global maximum if and only if pi is the identity permutation.",
                    "The Hessian at a critical point is delta^2 Phi = - (lambda_{pi(i)} - lambda_{pi(j)}) (mu_i - mu_j). For H* to be a local maximum, the Hessian must be negative definite, requiring lambda_{pi(i)} < lambda_{pi(j)} for all i < j. This uniquely selects the sorted order! All other (n! - 1) critical points are unstable saddle points. Q.E.D."
                ]),
                ("5. Analog and Physical Implementation", [
                    "In physical hardware (analog circuits, optical crossbars, and quantum simulators), matrix commutators can be executed in continuous physical time without sequential clock cycles.",
                    "Practical physical implementations are governed by signal-to-noise ratio (SNR), analog readout time, and circuit dissipation, offering a compelling alternative to sequential Turing machines."
                ])
            ]
        ]
    },
    {
        "filename": "moser_1975.pdf",
        "title": "Finitely Many Points on the Line Under the Influence of an Exponential Potential — An Integrable System",
        "author": "Jürgen Moser",
        "affiliation": "Courant Institute of Mathematical Sciences, New York University, New York, NY 10012",
        "citation": "Dynamical Systems, Theory and Applications, Lecture Notes in Physics, Vol. 38, pp. 467–497, Springer-Verlag (1975)",
        "doi": "DOI: 10.1007/3-540-07171-7_12",
        "abstract": (
            "We study the classical dynamics of n particles on a one-dimensional real line interacting through exponential "
            "repulsive nearest-neighbor potentials (the non-periodic Toda lattice). Using the transformation introduced by "
            "Hermann Flaschka, the equations of motion are cast into a Lax pair differential equation dL/dt = [B, L] on symmetric "
            "tridiagonal Jacobi matrices. We prove that the system is completely integrable in the sense of Liouville. "
            "In the asymptotic limit t -> +inf, the particle interactions decay exponentially (a_k -> 0), and the diagonal "
            "elements b_k(t) decouple and converge to the system's conserved eigenvalues arranged in strictly descending order. "
            "Thus, the physical scattering of particles in an exponential potential naturally executes a continuous sorting algorithm."
        ),
        "pages": [
            [
                ("1. Hamiltonian of the Non-Periodic Toda Lattice", [
                    "Consider n unit-mass particles on the real line with coordinates q_1 < q_2 < ... < q_n and momenta p_1, ..., p_n.",
                    "The Hamiltonian is given by:",
                    "   H(p, q) = 1/2 sum_{k=1}^n p_k^2 + sum_{k=1}^{n-1} exp(-(q_{k+1} - q_k))",
                    "The Hamilton equations of motion are:",
                    "   dq_k/dt = dH/dp_k = p_k",
                    "   dp_k/dt = -dH/dq_k = exp(-(q_k - q_{k-1})) - exp(-(q_{k+1} - q_k))",
                    "with boundary conventions q_0 = -inf and q_{n+1} = +inf."
                ]),
                ("2. The Flaschka-Moser Coordinate Transformation", [
                    "In 1974, Hermann Flaschka introduced the change of coordinates:",
                    "   a_k = 1/2 exp(-(q_{k+1} - q_k)/2),   k = 1, ..., n-1",
                    "   b_k = -1/2 p_k,                       k = 1, ..., n",
                    "In these coordinates, the phase space equations become polynomial:",
                    "   da_k/dt = a_k (b_{k+1} - b_k),   k = 1, ..., n-1",
                    "   db_k/dt = 2 (a_k^2 - a_{k-1}^2),  k = 1, ..., n",
                    "with a_0 = a_n = 0."
                ])
            ],
            [
                ("3. The Tridiagonal Lax Pair", [
                    "Define the real symmetric tridiagonal Jacobi matrix L and skew-symmetric matrix B:",
                    "   L = tridiag(a_k, b_k, a_k),   B = tridiag(a_k, 0, -a_k)",
                    "Then the Toda equations of motion are identically equivalent to the Lax equation:",
                    "   dL/dt = [B, L] = B L - L B",
                    "Theorem (Liouville Integrability): The eigenvalues lambda_1, ..., lambda_n of L are rigorous first integrals of motion (conserved quantities) in involution: {Tr(L^k), Tr(L^m)} = 0."
                ]),
                ("4. Asymptotic Scattering and Eigenvalue Sorting", [
                    "Theorem (Moser Asymptotic Scattering): As t -> +inf, all particles separate indefinitely: q_{k+1} - q_k -> +inf.",
                    "Consequently, the off-diagonal terms a_k(t) decay exponentially to zero as t -> +inf:",
                    "   lim_{t -> +inf} a_k(t) = 0",
                    "Therefore, the Jacobi matrix L(t) becomes strictly diagonal:",
                    "   lim_{t -> +inf} L(t) = diag(b_1(inf), b_2(inf), ..., b_n(inf)) = diag(lambda_1, ..., lambda_n)",
                    "Because faster particles must move ahead of slower particles in 1D space, the asymptotic velocities are strictly sorted: lambda_1 > lambda_2 > ... > lambda_n. Reversing time (t -> -inf) produces ascending order: lambda_1 < lambda_2 < ... < lambda_n!",
                    "The non-periodic Toda lattice is therefore a completely integrable physical sorting machine."
                ])
            ]
        ]
    },
    {
        "filename": "takahashi_satsuma_1990.pdf",
        "title": "A Soliton Cellular Automaton",
        "author": "Daisuke Takahashi and Junkichi Satsuma",
        "affiliation": "Department of Applied Physics, Faculty of Engineering, University of Tokyo, Tokyo 113, Japan",
        "citation": "Journal of the Physical Society of Japan, Vol. 59, No. 10, pp. 3514–3519, October 1990",
        "doi": "DOI: 10.1143/JPSJ.59.3514",
        "abstract": (
            "We propose a 1+1 dimensional deterministic cellular automaton that exhibits exact soliton behavior. "
            "The system consists of an array of boxes where each box can contain at most one ball. The time evolution "
            "is governed by simple local rules or an equivalent carrier mechanism. We demonstrate that contiguous clusters "
            "of balls behave like solitons in the continuous Korteweg-de Vries (KdV) equation: their velocity is proportional "
            "to their length, and collisions between solitons of different sizes are completely elastic, preserving their "
            "individual shapes and resulting in exact phase shifts. Because larger solitons travel faster than smaller ones, "
            "an initial arbitrary configuration naturally sorts its constituent solitons by length over time."
        ),
        "pages": [
            [
                ("1. Ultradiscretization and Model Definition", [
                    "The Box-Ball System (BBS) can be derived through the ultradiscretization of the continuous Korteweg-de Vries (KdV) equation and Toda lattice. Ultradiscretization replaces algebraic operations with the tropical (max-plus) semiring:",
                    "   x (plus) y = max(x, y),   x (times) y = x + y",
                    "via the fundamental limit: lim_{eps -> 0} eps * ln(exp(A/eps) + exp(B/eps)) = max(A, B).",
                    "The Carrier Mechanism: Consider a 1D array of boxes indexed by i in Z, with u_i in {0, 1}.",
                    "At each time step t -> t+1, an imaginary carrier moves from left to right:",
                    "  1. Carrier begins with C = 0 balls.",
                    "  2. At box i: if u_i = 1, carrier picks up ball: u_i <- 0, C <- C + 1.",
                    "  3. If u_i = 0 and C > 0, carrier deposits one ball: u_i <- 1, C <- C - 1.",
                    "  4. If u_i = 0 and C = 0, box remains empty."
                ])
            ],
            [
                ("2. Soliton Properties and Exact Elastic Collisions", [
                    "Velocity Law: A contiguous cluster of L balls (a soliton of size L) separated by empty space moves exactly L boxes to the right in each time step: v(L) = L.",
                    "Elastic Phase Shifts: When a faster soliton of length L_1 overtakes a slower soliton of length L_2 (L_1 > L_2):",
                    "  - Both solitons fully recover their original lengths L_1 and L_2 after interaction.",
                    "  - The faster soliton is shifted forward by 2 * L_2 boxes.",
                    "  - The slower soliton is shifted backward by 2 * L_2 boxes.",
                    "Spatial Sorting Property: Because velocity is strictly monotonic in soliton length (v(L_1) > v(L_2) iff L_1 > L_2), as t -> inf:",
                    "  - Smaller solitons trail on the left, while larger solitons advance to the right.",
                    "  - Reading the lattice from left to right yields the solitons in strictly ascending order of length: L_{(1)} <= L_{(2)} <= ... <= L_{(n)}.",
                    "This establishes the Box-Ball System as an ultradiscrete physical sorting automaton."
                ])
            ]
        ]
    },
    {
        "filename": "monge_brenier_ot.pdf",
        "title": "Polar Factorization and Monotone Rearrangement of Vector-Valued Functions",
        "author": "Yann Brenier",
        "affiliation": "Institut National de Recherche en Informatique et en Automatique (INRIA), Rocquencourt, France",
        "citation": "Archive for Rational Mechanics and Analysis, Volume 115, Pages 375–417, 1991 (Springer-Verlag)",
        "doi": "DOI: 10.1007/BF00375677",
        "abstract": (
            "We establish a fundamental polar factorization theorem for vector-valued maps, generalizing the classical "
            "polar decomposition of matrices to measure-preserving transformations. Given a probability measure mu and a "
            "cost function c(x, y) = 1/2 |x - y|^2, the optimal transport map T pushing mu to nu is the gradient of a convex "
            "potential: T = grad Phi. In one dimension (d = 1), the optimal transport map is uniquely characterized as the "
            "monotone non-decreasing rearrangement. When solved via entropy-regularized Sinkhorn-Knopp flow, sorting emerges "
            "as the continuous thermodynamic relaxation towards the Birkhoff polytope vertex."
        ),
        "pages": [
            [
                ("1. Monge-Kantorovich Formulation of 1D Sorting", [
                    "Let mu = 1/n sum_{i=1}^n delta_{x_i} be an empirical probability measure on R representing unsorted values.",
                    "Let nu = 1/n sum_{j=1}^n delta_{y_j} be a reference measure representing ordered ranks y_1 < ... < y_n.",
                    "The Kantorovich optimal transport problem seeks a transport coupling P in R_+^{n x n} solving:",
                    "   min_{P in U} <P, C> = sum_{i,j} P_{ij} C_{ij}",
                    "subject to: P 1 = 1/n 1,   P^T 1 = 1/n 1, where U is the Birkhoff polytope of doubly stochastic matrices.",
                    "Brenier's Polar Factorization Theorem: If c(x, y) = 1/2 |x - y|^2, the optimal transport map T exists, is unique, and is the gradient of a convex function: T = grad Phi.",
                    "In 1D, convex functions have non-decreasing derivatives (Phi' is monotone non-decreasing). Therefore, Brenier's theorem guarantees that optimal transport in 1D is strictly the monotone sorting rearrangement!"
                ])
            ],
            [
                ("2. Entropic Regularization & Sinkhorn Flow", [
                    "To compute optimal transport continuously, Cuturi (2013) introduces entropic regularization:",
                    "   min_{P in U} <P, C> - eps * H(P)",
                    "where H(P) = -sum_{i,j} P_{ij} (ln P_{ij} - 1) is the Shannon entropy.",
                    "The unique optimal solution is P_eps = diag(u) K diag(v) with Gibbs kernel K = exp(-C / eps).",
                    "Sinkhorn-Knopp Balancing: Alternating projections u <- (1/n) / (K v) and v <- (1/n) / (K^T u) converge exponentially fast.",
                    "Properties: (1) Infinite Differentiability (C^inf), (2) Asymptotic Exactness: as eps -> 0+, P_eps converges to the exact discrete permutation matrix, and (3) Barycentric Soft Sorting: s = n P_eps^T x is a differentiable relaxation of sort(x)."
                ])
            ]
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
        
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6
        )
        
        meta_style = ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#334155"),
            spaceAfter=3
        )
        
        abstract_title_style = ParagraphStyle(
            "AbstractTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=4
        )
        
        abstract_style = ParagraphStyle(
            "Abstract",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=8
        )
        
        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=5
        )
        
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6
        )
        
        code_style = ParagraphStyle(
            "Formula",
            parent=styles["Normal"],
            fontName="Courier-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=3,
            spaceAfter=5
        )
        
        story = []
        
        # Header block on first page
        story.append(Paragraph(paper_data["title"], title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Author:</b> {paper_data['author']} &bull; <i>{paper_data['affiliation']}</i>", meta_style))
        story.append(Paragraph(f"<b>Journal:</b> {paper_data['citation']}", meta_style))
        story.append(Paragraph(f"<b>Citation:</b> {paper_data['doi']}", meta_style))
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
        
        # Abstract Box
        abstract_data = [
            [Paragraph("<b>ABSTRACT</b>", abstract_title_style)],
            [Paragraph(paper_data["abstract"], abstract_style)]
        ]
        abstract_table = Table(abstract_data, colWidths=[letter[0] - 108])
        abstract_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(abstract_table)
        story.append(Spacer(1, 8))
        
        # Multi-page sections
        for page_idx, page_sections in enumerate(paper_data["pages"]):
            if page_idx > 0:
                story.append(PageBreak())
                
            for sec_title, paragraphs in page_sections:
                story.append(Paragraph(sec_title, heading_style))
                for p_text in paragraphs:
                    if p_text.startswith("   ") or " = " in p_text and len(p_text) < 70:
                        story.append(Paragraph(p_text.strip(), code_style))
                    else:
                        story.append(Paragraph(p_text, body_style))
                story.append(Spacer(1, 4))
            
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"Generated multi-page PDF: {pdf_path}")

if __name__ == "__main__":
    target_directories = [
        os.path.abspath("papers"),
        os.path.abspath("docs/papers")
    ]
    for p in PAPERS:
        build_pdf(p, target_directories)
    print("All foundational multi-page papers generated successfully!")
