import React from "react";
import { Redirect } from "react-router-dom"
import { Button } from "react-bootstrap"


export class Home extends React.Component {

    render() {

        return (
            <div className="mt-5 mx-5">
                <span className="font-weight-bold" style={{ fontSize: 20 }}>ChemRxnExtractor:</span> Chemical Reaction Extraction from Literature Text
                <div className="my-3">
                    <img src="chmextractor_diagram.png" className="img-fluid" alt="ChmExtractor Diagram" />
                </div>

                <h3>Introduction</h3>

                <p>ChemRxnExtractor (CRE) is a tool developed for automatically extracting chemical reactions in a structured format from scientific literature (journal articles).
                </p>

                <p>
                    CRE extracts major products of reactions and a set of associated reaction roles, including Reaction type, Reactants, Solvent, Temperature, etc.
                </p>

                <Button href="/extract" className="my-1"> > Try our online demo</Button> &nbsp; &nbsp; &nbsp; <Button href="https://pubs.acs.org/doi/10.1021/acs.jcim.1c00284"> > Read our paper</Button>

            </div>
        );
    }
}