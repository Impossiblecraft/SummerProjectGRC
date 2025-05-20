# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:28:16 2025

@author: Rajveer Daga
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Constants
# All units in kpc for distances and M_sun (solar masses) for mass
# Density will be in M_sun / kpc^3
# change constants


# Thin disk parameters (exponential disk)
# Values based on common Milky Way models
THIN_DISK_CENTRAL_DENSITY = 8.9e8  # M_sun / kpc^3
THIN_DISK_SCALE_RADIUS = 2.5  # kpc
THIN_DISK_SCALE_HEIGHT = 0.3  # kpc

# Thick disk parameters
THICK_DISK_CENTRAL_DENSITY = 1.83e8  # M_sun / kpc^3
THICK_DISK_SCALE_RADIUS = 3.02  # kpc
THICK_DISK_SCALE_HEIGHT = 0.9  # kpc

# Bulge parameters (Hernquist profile)
BULGE_MASS = 1.5e10  # M_sun
BULGE_SCALE_RADIUS = 0.7  # kpc

# Halo parameters (NFW profile)
HALO_CENTRAL_DENSITY = 8.1e6  # M_sun / kpc^3
HALO_SCALE_RADIUS = 16.0  # kpc

# HI gas disk parameters
HI_DISK_CENTRAL_DENSITY = 5.3e7  # M_sun / kpc^3
HI_DISK_SCALE_RADIUS = 7.0  # kpc
HI_DISK_SCALE_HEIGHT = 0.085  # kpc

# Baryonic halo parameters (hot gas and stellar halo)
# Based on typical Milky Way models - approximately 10% of DM halo is baryonic
# M_sun / kpc^3 (approximately 10% of DM halo)
BARY_HALO_CENTRAL_DENSITY = 8.1e5
BARY_HALO_SCALE_RADIUS = 18.0  # kpc


def thin_disk_density(r, z):
    """
    Calculate the mass density of the thin disk at a given cylindrical coordinate (r, z).

    Parameters:
    -----------
    r : float or array_like
        Galactocentric radius in kpc
    z : float or array_like
        Height above the galactic plane in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    return THIN_DISK_CENTRAL_DENSITY * np.exp(-r/THIN_DISK_SCALE_RADIUS) * np.exp(-np.abs(z)/THIN_DISK_SCALE_HEIGHT)


def thick_disk_density(r, z):
    """
    Calculate the mass density of the thick disk at a given cylindrical coordinate (r, z).

    Parameters:
    -----------
    r : float or array_like
        Galactocentric radius in kpc
    z : float or array_like
        Height above the galactic plane in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    return THICK_DISK_CENTRAL_DENSITY * np.exp(-r/THICK_DISK_SCALE_RADIUS) * np.exp(-np.abs(z)/THICK_DISK_SCALE_HEIGHT)


def total_disk_density(r, z):
    """
    Calculate the combined mass density of the thin and thick disks at a given cylindrical coordinate (r, z).

    Parameters:
    -----------
    r : float or array_like
        Galactocentric radius in kpc
    z : float or array_like
        Height above the galactic plane in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    return thin_disk_density(r, z) + thick_disk_density(r, z)


def hi_gas_disk_density(r, z):
    """
    Calculate the mass density of the HI gas disk at a given cylindrical coordinate (r, z).

    Parameters:
    -----------
    r : float or array_like
        Galactocentric radius in kpc
    z : float or array_like
        Height above the galactic plane in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    return HI_DISK_CENTRAL_DENSITY * np.exp(-r/HI_DISK_SCALE_RADIUS) * np.exp(-np.abs(z)/HI_DISK_SCALE_HEIGHT)


def bulge_density(r_spherical):
    """
    Calculate the mass density of the bulge using a Hernquist profile at a given spherical radius.

    Parameters:
    -----------
    r_spherical : float or array_like
        Spherical radius in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    # Hernquist profile
    return (BULGE_MASS / (2 * np.pi)) * (BULGE_SCALE_RADIUS / r_spherical) * (1 / (r_spherical + BULGE_SCALE_RADIUS)**3)


def halo_density(r_spherical):
    """
    Calculate the mass density of the dark matter halo using NFW profile at a given spherical radius.

    Parameters:
    -----------
    r_spherical : float or array_like
        Spherical radius in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    # NFW profile
    return HALO_CENTRAL_DENSITY / ((r_spherical / HALO_SCALE_RADIUS) * (1 + r_spherical / HALO_SCALE_RADIUS)**2)


def baryonic_halo_density(r_spherical):
    """
    Calculate the mass density of the baryonic halo (hot gas and stellar halo)
    using an NFW-like profile at a given spherical radius.

    Parameters:
    -----------
    r_spherical : float or array_like
        Spherical radius in kpc

    Returns:
    --------
    density : float or array_like
        Mass density in M_sun / kpc^3
    """
    # NFW-like profile for baryonic halo
    return BARY_HALO_CENTRAL_DENSITY / ((r_spherical / BARY_HALO_SCALE_RADIUS) * (1 + r_spherical / BARY_HALO_SCALE_RADIUS)**2)

# Coordinate conversion functions


def cylindrical_to_spherical(r, z):
    """
    Convert cylindrical coordinates (r, z) to spherical radius.

    Parameters:
    -----------
    r : float or array_like
        Cylindrical radius
    z : float or array_like
        Height

    Returns:
    --------
    spherical_r : float or array_like
        Spherical radius
    """
    return np.sqrt(r**2 + z**2)


def spherical_to_cylindrical(r_spherical, theta):
    """
    Convert spherical coordinates (r, theta) to cylindrical coordinates (r, z).

    Parameters:
    -----------
    r_spherical : float or array_like
        Spherical radius
    theta : float or array_like
        Polar angle (theta=0 at z-axis)

    Returns:
    --------
    r_cylindrical : float or array_like
        Cylindrical radius
    z : float or array_like
        Height
    """
    r_cylindrical = r_spherical * np.sin(theta)
    z = r_spherical * np.cos(theta)
    return r_cylindrical, z

# Integration functions

# Cylindrical coordinate integrands


def thin_disk_mass_integrand_cylindrical(z, r, phi):
    """
    Integrand function for thin disk mass in cylindrical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    z : float
        Height coordinate
    r : float
        Radial coordinate
    phi : float
        Azimuthal angle

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    return thin_disk_density(r, z) * r  # r factor from cylindrical volume element


def thick_disk_mass_integrand_cylindrical(z, r, phi):
    """
    Integrand function for thick disk mass in cylindrical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    z : float
        Height coordinate
    r : float
        Radial coordinate
    phi : float
        Azimuthal angle

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    return thick_disk_density(r, z) * r  # r factor from cylindrical volume element


def hi_disk_mass_integrand_cylindrical(z, r, phi):
    """
    Integrand function for HI gas disk mass in cylindrical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    z : float
        Height coordinate
    r : float
        Radial coordinate
    phi : float
        Azimuthal angle

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    return hi_gas_disk_density(r, z) * r  # r factor from cylindrical volume element

# Spherical coordinate integrands


def thin_disk_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for thin disk mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    r_cylindrical = r_spherical * np.sin(theta)  # Convert to cylindrical r
    z = r_spherical * np.cos(theta)              # Convert to cylindrical z

    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return thin_disk_density(r_cylindrical, z) * r_spherical**2 * np.sin(theta)


def thick_disk_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for thick disk mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    r_cylindrical = r_spherical * np.sin(theta)  # Convert to cylindrical r
    z = r_spherical * np.cos(theta)              # Convert to cylindrical z

    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return thick_disk_density(r_cylindrical, z) * r_spherical**2 * np.sin(theta)


def hi_disk_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for HI gas disk mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    r_cylindrical = r_spherical * np.sin(theta)  # Convert to cylindrical r
    z = r_spherical * np.cos(theta)              # Convert to cylindrical z

    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return hi_gas_disk_density(r_cylindrical, z) * r_spherical**2 * np.sin(theta)


def bulge_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for bulge mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return bulge_density(r_spherical) * r_spherical**2 * np.sin(theta)


def halo_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for halo mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return halo_density(r_spherical) * r_spherical**2 * np.sin(theta)


def baryonic_halo_mass_integrand_spherical(phi, theta, r_spherical):
    """
    Integrand function for baryonic halo mass in spherical coordinates.
    Note: Order of parameters is important for scipy.integrate

    Parameters:
    -----------
    phi : float
        Azimuthal angle
    theta : float
        Polar angle (theta=0 at z-axis)
    r_spherical : float
        Spherical radius

    Returns:
    --------
    float
        Value of the integrand at the specified point
    """
    # Include both sin(theta) from Jacobian and r_spherical^2 from volume element
    return baryonic_halo_density(r_spherical) * r_spherical**2 * np.sin(theta)


def calculate_thin_disk_mass_spherical(max_radius):
    """
    Calculate the mass of the thin disk up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the thin disk in solar masses
    """
    # Define the integration limits
    r_limits = [0, max_radius]
    theta_limits = [0, np.pi]  # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]  # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        thin_disk_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_thick_disk_mass_spherical(max_radius):
    """
    Calculate the mass of the thick disk up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the thick disk in solar masses
    """
    # Define the integration limits
    r_limits = [0, max_radius]
    theta_limits = [0, np.pi]  # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]  # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        thick_disk_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_hi_disk_mass_spherical(max_radius):
    """
    Calculate the mass of the HI gas disk up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the HI gas disk in solar masses
    """
    # Define the integration limits
    r_limits = [0, max_radius]
    theta_limits = [0, np.pi]  # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]  # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        hi_disk_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_bulge_mass_spherical(max_radius):
    """
    Calculate the mass of the bulge up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the bulge in solar masses
    """
    # Define the integration limits
    r_limits = [0.001, max_radius]  # Avoid r=0 for Hernquist profile
    theta_limits = [0, np.pi]       # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]       # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        bulge_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_halo_mass_spherical(max_radius):
    """
    Calculate the mass of the dark matter halo up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the halo in solar masses
    """
    # Define the integration limits
    r_limits = [0.001, max_radius]  # Avoid r=0 for NFW profile
    theta_limits = [0, np.pi]       # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]       # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        halo_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_baryonic_halo_mass_spherical(max_radius):
    """
    Calculate the mass of the baryonic halo up to a maximum radius using spherical coordinates.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    float
        Mass of the baryonic halo in solar masses
    """
    # Define the integration limits
    r_limits = [0.001, max_radius]  # Avoid r=0 for NFW profile
    theta_limits = [0, np.pi]       # Full range of theta (0 to π)
    phi_limits = [0, 2*np.pi]       # Full range of phi (0 to 2π)

    # Perform the triple integration
    result, error = integrate.nquad(
        baryonic_halo_mass_integrand_spherical,
        [phi_limits, theta_limits, r_limits]
    )

    return result


def calculate_total_baryonic_mass(max_radius):
    """
    Calculate the total baryonic mass of the Milky Way up to a specified radius.
    Includes thin disk, thick disk, HI gas disk, bulge, and baryonic halo.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    dict
        Dictionary containing the mass components and total mass in solar masses
    """
    thin_disk_mass = calculate_thin_disk_mass_spherical(max_radius)
    thick_disk_mass = calculate_thick_disk_mass_spherical(max_radius)
    hi_disk_mass = calculate_hi_disk_mass_spherical(max_radius)
    bulge_mass = calculate_bulge_mass_spherical(max_radius)
    bary_halo_mass = calculate_baryonic_halo_mass_spherical(max_radius)

    # Sum of all baryonic components (excluding dark matter halo)
    total_baryonic_mass = thin_disk_mass + thick_disk_mass + \
        hi_disk_mass + bulge_mass + bary_halo_mass

    return {
        'thin_disk_mass': thin_disk_mass,
        'thick_disk_mass': thick_disk_mass,
        'hi_disk_mass': hi_disk_mass,
        'bulge_mass': bulge_mass,
        'bary_halo_mass': bary_halo_mass,
        'total_baryonic_mass': total_baryonic_mass
    }


def calculate_total_mass_with_dm(max_radius):
    """
    Calculate the total mass (baryonic + dark matter) of the Milky Way up to a specified radius.

    Parameters:
    -----------
    max_radius : float
        Maximum radius for the integration in kpc

    Returns:
    --------
    dict
        Dictionary containing the mass components and total mass in solar masses
    """
    baryonic_data = calculate_total_baryonic_mass(max_radius)
    halo_mass = calculate_halo_mass_spherical(max_radius)

    total_mass = baryonic_data['total_baryonic_mass'] + halo_mass

    return {
        'thin_disk_mass': baryonic_data['thin_disk_mass'],
        'thick_disk_mass': baryonic_data['thick_disk_mass'],
        'hi_disk_mass': baryonic_data['hi_disk_mass'],
        'bulge_mass': baryonic_data['bulge_mass'],
        'bary_halo_mass': baryonic_data['bary_halo_mass'],
        'halo_mass': halo_mass,
        'total_baryonic_mass': baryonic_data['total_baryonic_mass'],
        'total_mass': total_mass
    }


"""
# Example usage
if __name__ == "__main__":
    # Calculate mass within different radii
    radii = [5, 10, 15, 20, 25]
    baryonic_masses = []
    total_masses = []
    component_masses = {
        'thin_disk': [],
        'thick_disk': [],
        'hi_disk': [],
        'bulge': [],
        'bary_halo': [],
        'halo': []
    }

    for radius in radii:
        print(f"Calculating mass within {radius} kpc...")
        mass_data = calculate_total_mass_with_dm(radius)
        baryonic_masses.append(mass_data['total_baryonic_mass'])
        total_masses.append(mass_data['total_mass'])

        # Store component masses for plotting
        component_masses['thin_disk'].append(mass_data['thin_disk_mass'])
        component_masses['thick_disk'].append(mass_data['thick_disk_mass'])
        component_masses['hi_disk'].append(mass_data['hi_disk_mass'])
        component_masses['bulge'].append(mass_data['bulge_mass'])
        component_masses['bary_halo'].append(mass_data['bary_halo_mass'])
        component_masses['halo'].append(mass_data['halo_mass'])

        print(f"  Thin disk: {mass_data['thin_disk_mass']:.2e} M_sun")
        print(f"  Thick disk: {mass_data['thick_disk_mass']:.2e} M_sun")
        print(f"  HI gas disk: {mass_data['hi_disk_mass']:.2e} M_sun")
        print(f"  Bulge: {mass_data['bulge_mass']:.2e} M_sun")
        print(f"  Baryonic Halo: {mass_data['bary_halo_mass']:.2e} M_sun")
        print(f"  Dark Matter Halo: {mass_data['halo_mass']:.2e} M_sun")
        print(f"  Total Baryonic: {
              mass_data['total_baryonic_mass']:.2e} M_sun")
        print(f"  Total Mass (with DM): {mass_data['total_mass']:.2e} M_sun")
        print("-" * 60)

    # Plot the mass as a function of radius
    plt.figure(figsize=(12, 8))

    # Plot individual components
    plt.plot(radii, component_masses['thin_disk'], 'o--', label='Thin Disk')
    plt.plot(radii, component_masses['thick_disk'], 's--', label='Thick Disk')
    plt.plot(radii, component_masses['hi_disk'], '^--', label='HI Gas Disk')
    plt.plot(radii, component_masses['bulge'], 'D--', label='Bulge')
    plt.plot(radii, component_masses['bary_halo'],
             'X--', label='Baryonic Halo')
    plt.plot(radii, component_masses['halo'], '*--', label='Dark Matter Halo')

    # Plot total masses
    plt.plot(radii, baryonic_masses, 'o-',
             linewidth=2, label='Total Baryonic Mass')
    plt.plot(radii, total_masses, 's-', linewidth=2,
             label='Total Mass (with DM)')

    plt.xlabel('Radius (kpc)')
    plt.ylabel('Enclosed Mass (M_sun)')
    plt.title('Milky Way Mass Components vs. Radius')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.savefig('milky_way_mass_components.png')
    plt.show()
"""
