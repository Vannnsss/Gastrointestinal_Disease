// toggle class hamburger active
const navbarNav = document.querySelector(".navbar-nav");
document.querySelector("#hamburger-menu").onclick = () => {
  navbarNav.classList.toggle("active");
};

// toggle class nav active remove
const navb = document.querySelector(".navbar-nav");
document.addEventListener("click", function (e) {
  if (navb.contains(e.target)) {
    navb.classList.remove("active");
  }
});


