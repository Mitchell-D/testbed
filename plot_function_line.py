import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


def plot_line(x, y, plot_spec={}, fig_path=None, show=False):
    """
    overwhelmingly basic line plotting method, for when you
    just need a dang sine curve or something.
    """
    # Plot the data
    fig,ax = plt.subplots()
    ax.plot(x, y)

    # Set the axis plot_spec and title using the dictionary
    ax.set_xlabel(
            plot_spec.get("xlabel", ""),
            fontsize=plot_spec.get("font_size", 14),
            )
    ax.set_ylabel(
            plot_spec.get("ylabel", ""),
            fontsize=plot_spec.get("font_size", 14),
            )
    ax.set_title(
            plot_spec.get("title", ""),
            fontsize=plot_spec.get("font_size", 16),
            )
    if plot_spec.get("grid"):
        ax.grid()
        ax.axhline(0, color='black')

    # Show the plot
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(
                fig_path.as_posix(),
                dpi=plot_spec.get("dpi", 200),
                bbox_inches="tight",
                )

if __name__=="__main__":
    # Example usage

    x = np.linspace(0, 1., 4000)
    y = -1 * x * np.log(x)
    plot_spec = {
            "title": "Integrand of Entropy $\\frac{d(H(X))}{dx} = -P(x)\,\ln(P(x))$",
            "xlabel": "Probability",
            "ylabel": "Entropy (nats)",
            "grid":True,
            }
    plot_line(x, y, plot_spec, show=True, fig_path=None)
