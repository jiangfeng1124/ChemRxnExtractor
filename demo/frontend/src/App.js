import React from "react"
import {
  BrowserRouter as Router,
  Switch,
  Route
} from "react-router-dom"
import { Navbar, Nav } from "react-bootstrap";
import { Extract } from "./components/Extract";
import { Home } from "./components/Home"
import { Download } from "./components/Download"
// import { Upload } from "./components/Upload"
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  return (
    <div className="App">
      <Router>
        <div>
          <Navbar bg="dark" variant="dark" className="px-2">
            <Navbar.Brand href="/home">ChemRxnExtractor</Navbar.Brand>
            <Nav className="mr-auto">
              <Nav.Link href="/home">Home</Nav.Link>
              <Nav.Link href="/extract">Online Demo</Nav.Link>
              <Nav.Link href="/download">Download</Nav.Link>
            </Nav>
          </Navbar>

          <Switch>

            <Route path="/extract">
              <Extract />
            </Route>

            <Route path="/download">
              <Download />
            </Route>

            <Route path="/">
              <Home />
            </Route>

          </Switch>
        </div>
      </Router>

      <footer class="footer">
        <div className="container">
          <span class="footer-head">ChemRxnExtractor v0.1 @ 2021</span> &nbsp; <a href="https://accessibility.mit.edu/">Accessibility</a> <br />

          <p class="footer-info">
            Computer Science and Artificial Intelligence Laboratory <br />
            Department of Chemical Engineering <br />
            Massachusetts Institute of Technology
          </p>
        </div>
      </footer>
    </div >
  );
}

export default App;
