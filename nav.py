import streamlit as st
from streamlit_option_menu import option_menu


PAGES = {
    "SmartTwin Warehouse": "app.py",
    "Waste prediction": "pages/waste.py",
    "Scenarios": "pages/scenarios.py",
    "Operational Intelligence": "pages/operational-inteligence.py"
}

def top_nav(
    active: str = "Overview",
    logo_url: str = "https://upload.wikimedia.org/wikipedia/de/d/de/Gategroup_Holding_201x_logo.svg",
    logo_width: int = 350,
):
    # ---- CSS del navbar
    st.markdown(
        """
        <style>
          /* fila contenedora */
          .gg-row {
            border-bottom: 1px solid rgba(0,0,0,.08);
            padding: 8px 16px 10px 16px;
            margin-bottom: 8px;
          }

          /* fuerza el option_menu a la derecha y en línea */
          .gg-right .container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
          .gg-right .nav { display: flex !important; justify-content: flex-end !important; width: 100%; }
          .gg-right .nav-link { font-weight: 600; padding: 6px 14px; border-radius: 999px; }
          .gg-right .nav-link:hover { background: rgba(0,1,100,.06); }
          .gg-right .nav-link.active { background: #000164 !important; color: #fff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Contenedor con columnas (logo izq, menú der)
    st.markdown('<div class="gg-row">', unsafe_allow_html=True)
    left, right = st.columns([1, 3], vertical_alignment="center")

    with left:
        # Usa st.image para evitar problemas de SVG/HTML
        st.image("assets/gategroup.svg", width=logo_width)

    with right:
        st.markdown('<div class="gg-right">', unsafe_allow_html=True)
        labels = list(PAGES.keys())
        default_index = labels.index(active) if active in labels else 0
        selected = option_menu(
            None,
            labels,
            icons=["boxes", "recycle", "activity", "cpu"],
            orientation="horizontal",
            default_index=default_index,
            styles={"container": {"padding": "0", "margin": "0"},
                    "nav": {"padding": "0", "margin": "0"}}
        )
        st.markdown("</div>", unsafe_allow_html=True)  # cierra .gg-right
    st.markdown("</div>", unsafe_allow_html=True)      # cierra .gg-row

    # ---- Navegación
    target = PAGES[selected]
    if selected != active:
        if hasattr(st, "switch_page"):
            st.switch_page(target)
        else:
            st.page_link(target, label=f"Open {selected} →")
